from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import torch
import torch.nn as nn
import warnings
from contextlib import nullcontext

from user_model import UserModel
from trl import DPOTrainer
from transformers import PreTrainedModel


class UserDPOTrainer(DPOTrainer):
    def __init__(self, alpha, sep, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        # the same as self.sep_id = self.tokenizer(sep)["input_ids"][-1]
        self.sep_id = self.tokenizer.sep_token_id

    def _get_user_ids(
        self, 
        batch: Dict[str, Union[List, torch.LongTensor]],
    ):
        input_ids = batch["chosen_input_ids"]  
        sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
        # tokenized user_identifier are between the beginning bos token and the sep token
        user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]
        user_ids = UserModel.get_user_ids(self.tokenizer, user_input_ids) # (batch_size, )
        return user_ids

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
        return_mean: bool = True,
    ):
        """Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
        metrics = {}

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            nll_loss
        ) = self.concatenated_forward(model, batch)

        # if reference_chosen_logps and reference_rejected_logps in batch use them, otherwise use the reference model
        if "reference_chosen_logps" in batch and "reference_rejected_logps" in batch:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            with torch.no_grad():
                if self.ref_model is None:
                    with self.null_ref_context():
                        (
                            reference_chosen_logps,
                            reference_rejected_logps,
                            _,
                            _,
                            _,
                        ) = self.concatenated_forward(self.model, batch)
                else:
                    (
                        reference_chosen_logps,
                        reference_rejected_logps,
                        _,
                        _,
                        _,
                    ) = self.concatenated_forward(self.ref_model, batch)

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        user_ids = self._get_user_ids(batch)

        # gather data across all devices
        user_ids = self.accelerator.gather_for_metrics((user_ids))
        chosen_rewards = self.accelerator.gather_for_metrics((chosen_rewards))
        rejected_rewards = self.accelerator.gather_for_metrics((rejected_rewards))
        policy_chosen_logps = self.accelerator.gather_for_metrics((policy_chosen_logps))
        policy_rejected_logps = self.accelerator.gather_for_metrics((policy_rejected_logps))
        
        reward_accuracies = (chosen_rewards > rejected_rewards).float()  # (batch_size, )
        logp_accuracies = (policy_chosen_logps.detach() > policy_rejected_logps.detach()).float() # (batch_size, ) 

        prefix = "eval_" if train_eval == "eval" else ""
        for i in torch.unique(user_ids):  # user_id = 0 is the generic user
            metrics[f"{prefix}rewards/user_{i:02}_accuracies"] = reward_accuracies[user_ids == i].cpu()
            metrics[f"{prefix}logps/user_{i:02}_accuracies"] = logp_accuracies[user_ids == i].cpu()
            metrics[f"{prefix}logps/user_{i:02}_chosen"] = policy_chosen_logps[user_ids == i].detach().cpu()
            metrics[f"{prefix}logps/user_{i:02}_rejected"] = policy_rejected_logps[user_ids == i].detach().cpu()
        
        if (user_ids != 0).sum() > 0:
            metrics[f"{prefix}rewards/user_each_accuracies"] = reward_accuracies[user_ids != 0].cpu()
            metrics[f"{prefix}logps/user_each_accuracies"] = logp_accuracies[user_ids != 0].cpu()
            metrics[f"{prefix}logps/user_each_chosen"] = policy_chosen_logps[user_ids != 0].detach().cpu()
            metrics[f"{prefix}logps/user_each_rejected"] = policy_rejected_logps[user_ids != 0].detach().cpu()

        if return_mean:
            return losses.mean(), metrics
        else:
            return losses, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior. Alternatively, you can implement your own prediction_step method if you are using a custom data collator"
            )

        user_ids = self._get_user_ids(inputs)

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        # split the losses into generic_user (user_id = 0) and individual user (user_id > 0)
        with compute_loss_context_manager():
            losses, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train", return_mean=False)
            loss = 0
            if (user_ids != 0).sum() > 0:
                loss += self.alpha * losses[user_ids != 0].mean()
            if (user_ids == 0).sum() > 0:
                loss += (1. - self.alpha) * losses[user_ids == 0].mean()

        # force log the metrics
        if self.accelerator.is_main_process:
            self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss
    
    def log(self, logs: Dict[str, float]) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        sorted_keys = sorted(list(self._stored_metrics[train_eval].keys()))
        for key in sorted_keys:
            metrics = self._stored_metrics[train_eval][key]
            logs[key] = torch.concatenate(metrics).mean().item()  # batch

        del self._stored_metrics[train_eval]
        return super().log(logs)
    