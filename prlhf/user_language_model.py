from typing import Optional, List, Union, Tuple
import numpy as np
import torch
from transformers import (
    GPTNeoForCausalLM,
    GPTJForCausalLM,
    PreTrainedTokenizerBase,
    LlamaForCausalLM,
)
from user_model import IndividualUserModel, ClusterUserModel


class UserGPTJMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        user_model_type: str = "individual",
        tokenizer: PreTrainedTokenizerBase = None,
        n_users: int = 0,
        n_clusters: Optional[int] = None,
        n_user_tokens: int = 1,
        seed: int = 123, 
        add_generic_user: Optional[bool] = True,
        initialize_from_vocab: bool = True,
        most_common_tokens: torch.tensor = None,
        random_range: float = 0.5,
        sep: str = '|',
        is_reference: bool = False,
        **kwargs,
    ):
        """
        Initialize a user conditioned language model from a pretrained LLM. 

        Args:
            pretrained_model_name_or_path: The path to the pretrained LLM. 
            user_model_type: The type of user model to initialize, has to be either "individual" or "cluster".
            tokenizer: The tokenizer to use in the user model.
            n_users: Number of known users.
            n_clusters: Number of clusters in user model with cluster-based preference.
            n_user_tokens: User token length T_u.
            add_generic_user: Whether to add the generic implicit user embeddings to individual implicit user
                embeddings in user model with individualized preference.
            initialize_from_vocab: If True, embeddings in user model will be initialized from word embeddings 
                of the base LLM, otherwise will be initialized randomly.
            most_common_tokens: If provided and initialize_from_vocab = True, embeddings in user model will be 
                initialized from the word embeddings of the most common tokens. If most_common_tokens = None and 
                initialize_from_vocab = True, embeddings in user model will be initialized from randomly sampled
                word embeddings.
            random_range: If initialize_from_vocab = False, embeddings in user model will be initialized randomly 
                from the uniform distribution [-random_range, random_range].
            sep: The separator added between the user identifier and the text prompt. 
            is_reference: Whether the initialized LLM will be used as the reference model (non-personalized) or not.
            kwargs: keyword arguments for the LLM.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.config.n_embd = model.config.hidden_size if hasattr(model.config, "hidden_size") else model.config.n_embd

        assert user_model_type is not None, "user_model_type must be specified!"
        model.initialize_user_model(user_model_type, tokenizer, n_users, n_clusters, n_user_tokens,
                                    seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range)

        model.config.pad_token_id = model.user_model.tokenizer.eos_token_id
        model.is_reference = is_reference
        model.sep_id = model.user_model.tokenizer(sep)["input_ids"][0]
        return model
    
    def initialize_user_model(self, user_model_type, tokenizer, n_users, n_clusters, n_user_tokens, 
                              seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range) -> None:
        if user_model_type not in  ["individual", "cluster"]:
            raise NotImplementedError("Only individual and cluster user models are implemented!")

        n_init_tokens = n_users + 1 if user_model_type == "individual" else n_clusters
        n_init_tokens = n_init_tokens * n_user_tokens
        if initialize_from_vocab:
            if most_common_tokens is None: 
                most_common_tokens = self.transformer.wte.weight.size(0)
            elif isinstance(most_common_tokens, torch.Tensor):
                most_common_tokens = most_common_tokens.detach().data.numpy()

            np.random.seed(seed)
            init_tokens = torch.from_numpy(
                np.random.choice(most_common_tokens, size=n_init_tokens, replace=False)
            ).to(self.device)
            init_value = self.transformer.wte(init_tokens).clone().detach()
        else:
            init_value = None

        # user_embed_dim = base LLM embedding dim 
        if user_model_type == "individual":
            self.user_model = IndividualUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.n_embd, 
                n_users=n_users, 
                n_user_tokens=n_user_tokens, 
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
                add_generic_user=add_generic_user,
            )
        elif user_model_type == "cluster":
            self.user_model = ClusterUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.n_embd,
                n_users=n_users, 
                n_clusters=n_clusters, 
                n_user_tokens=n_user_tokens,
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
            )
        print(f"Initialized {user_model_type} user model!")

    def _cat_user_embedding_to_input_sep(self, input_ids, attention_mask):
        # for DPO trainer, input_ids: bos token + user_identifier + sep + prompt + response + eos token + padding
        # all sequences in the same batch are padded to the max sequence length in the batch
        # (batch_size, max_len, n_embed) or (max_len, n_embed)
        inputs_embeds = self.transformer.wte(input_ids) 
        
        if len(list(inputs_embeds.shape)) == 2:  # (max_len, n_embed)
            # (batch_size=1, max_len, n_embed) 
            inputs_embeds = inputs_embeds.unsqueeze(dim=0)

        if len(list(attention_mask.shape)) == 1:  # (max_len,)
            # (batch_size=1, max_len)
            attention_mask = attention_mask.unsqueeze(dim=0)

        # Get user_embeddings
        sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
        # tokenized user_identifier are between the beginning bos token and the sep token
        user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]
        # (batch_size, n_user_tokens, user_embed_dim)
        user_embeddings = self.user_model.get_user_embeddings(user_input_ids, generic_user=False) 

        if len(list(user_embeddings.shape)) == 1:  # (user_embed_dim, )
            user_embeddings = user_embeddings.unsqueeze(dim=0) # (batch_size=1, user_embed_dim)
        if len(list(user_embeddings.shape)) == 2:  # (batch_size, user_embed_dim)
            user_embeddings = user_embeddings.unsqueeze(dim=1) # (batch_size, n_user_tokens=1, user_embed_dim)

        new_input_embeds = []
        new_attention_masks = []

        for row_idx, row in enumerate(inputs_embeds):  # (batch_size, max_len, n_embed)
            user_embedding = user_embeddings[row_idx, :] if self.is_reference == False else \
                torch.full((0, user_embeddings[row_idx, :].size(1)), fill_value=0, dtype=row.dtype, 
                           device=user_embeddings.device)  # (n_user_tokens, user_embed_dim)
            pad_input_ids = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),),  
                                       fill_value=self.user_model.tokenizer.pad_token_id, 
                                       dtype=input_ids.dtype, 
                                       device=input_ids.device)
            pad_input_embeds = self.transformer.wte(pad_input_ids)
            # pad to the left so that the labels are valid when computing logps in DPO trainer
            # concatenate [padding for user_identifier, bos token, user_embedding, prompt + response + eos token + padding]
            new_input_embeds.append(torch.concat(
                [pad_input_embeds, row[:1, :], user_embedding, row[(sep_indices[row_idx]+1):, :]], 
                dim=0)
            )
            
            attention_mask_row = attention_mask[row_idx]  # (max_len,)
            user_attention_mask = torch.full(size=(user_embedding.size(0),), 
                                             fill_value=1, 
                                             device=attention_mask.device)  # (user_tokens,)
            pad_attention_mask = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),), 
                                            fill_value=0, 
                                            device=attention_mask.device)
            new_attention_masks.append(torch.concat(
                [pad_attention_mask, attention_mask_row[:1], user_attention_mask,
                 attention_mask_row[(sep_indices[row_idx]+1):]], 
                 dim=0)
            )
        
        # (batch_size, max_len, user_embed_dim)  
        inputs_embeds = torch.stack(new_input_embeds, dim=0).to(input_ids.device)
        attention_mask = torch.stack(new_attention_masks, dim=0).to(attention_mask.device)

        return inputs_embeds, attention_mask, user_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ): 
        # For DPO trainer, a separator is added between user identifier and the prompt, and then tokenized.
        inputs_embeds, attention_mask, user_embeddings = self._cat_user_embedding_to_input_sep(
            input_ids=input_ids, attention_mask=attention_mask)
        
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels, 
            **kwargs,
        )


class UserGPTNeoForCausalLM(UserGPTJMixin, GPTNeoForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class UserGPTJForCausalLM(UserGPTJMixin, GPTJForCausalLM):
    def __init__(self, config):
        super().__init__(config)


class UserLlamaMixin:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        user_model_type: str = "individual",
        tokenizer: PreTrainedTokenizerBase = None,
        n_users: int = 0,
        n_clusters: Optional[int] = None,
        n_user_tokens: int = 1,
        seed: int = 123, 
        add_generic_user: Optional[bool] = True,
        initialize_from_vocab: bool = True,
        most_common_tokens: torch.tensor = None,
        random_range: float = 0.5,
        sep: str = '|',
        is_reference: bool = False,
        **kwargs,
    ):
        """
        Initialize a user conditioned language model from a pretrained LLM. 

        Args:
            pretrained_model_name_or_path: The path to the pretrained LLM. 
            user_model_type: The type of user model to initialize, has to be either "individual" or "cluster".
            tokenizer: The tokenizer to use in the user model.
            n_users: Number of known users.
            n_clusters: Number of clusters in user model with cluster-based preference.
            n_user_tokens: User token length T_u.
            add_generic_user: Whether to add the generic implicit user embeddings to individual implicit user
                embeddings in user model with individualized preference.
            initialize_from_vocab: If True, embeddings in user model will be initialized from word embeddings 
                of the base LLM, otherwise will be initialized randomly.
            most_common_tokens: If provided and initialize_from_vocab = True, embeddings in user model will be 
                initialized from the word embeddings of the most common tokens. If most_common_tokens = None and 
                initialize_from_vocab = True, embeddings in user model will be initialized from randomly sampled
                word embeddings.
            random_range: If initialize_from_vocab = False, embeddings in user model will be initialized randomly 
                from the uniform distribution [-random_range, random_range].
            sep: The separator added between the user identifier and the text prompt. 
            is_reference: Whether the initialized LLM will be used as the reference model (non-personalized) or not.
            kwargs: keyword arguments for the LLM.
        """
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        assert user_model_type is not None, "user_model_type must be specified!"
        model.initialize_user_model(user_model_type, tokenizer, n_users, n_clusters, n_user_tokens,
                                    seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range)

        model.config.pad_token_id = model.user_model.tokenizer.eos_token_id
        model.is_reference = is_reference
        model.sep_id = model.user_model.tokenizer(sep)["input_ids"][-1]
        return model
    
    def initialize_user_model(self, user_model_type, tokenizer, n_users, n_clusters, n_user_tokens, 
                              seed, add_generic_user, initialize_from_vocab, most_common_tokens, random_range) -> None:
        if user_model_type not in  ["individual", "cluster"]:
            raise NotImplementedError("Only individual and cluster user models are implemented!")

        n_init_tokens = n_users + 1 if user_model_type == "individual" else n_clusters
        n_init_tokens = n_init_tokens * n_user_tokens
        if initialize_from_vocab:
            if most_common_tokens is None: 
                most_common_tokens = self.model.embed_tokens.weight.size(0)
            elif isinstance(most_common_tokens, torch.Tensor):
                most_common_tokens = most_common_tokens.detach().data.numpy()

            np.random.seed(seed)
            init_tokens = torch.from_numpy(
                np.random.choice(most_common_tokens, size=n_init_tokens, replace=False)
            ).to(self.device)
            init_value = self.model.embed_tokens(init_tokens).clone().detach()
        else:
            init_value = None

        # user_embed_dim = base LLM embedding dim 
        if user_model_type == "individual":
            self.user_model = IndividualUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.hidden_size, 
                n_users=n_users, 
                n_user_tokens=n_user_tokens, 
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
                add_generic_user=add_generic_user,
            )
        elif user_model_type == "cluster":
            self.user_model = ClusterUserModel(
                tokenizer=tokenizer, 
                user_embed_dim=self.config.hidden_size,
                n_users=n_users, 
                n_clusters=n_clusters, 
                n_user_tokens=n_user_tokens,
                init_value=init_value, 
                random_range=random_range, 
                seed=seed,
            )
        print(f"Initialized {user_model_type} user model!")

    def _cat_user_embedding_to_input_sep(self, input_ids, attention_mask):
        # for DPO trainer, input_ids: bos token + user_identifier + sep + prompt + response + eos token + padding
        # all sequences in the same batch are padded to the max sequence length in the batch
        # (batch_size, max_len, n_embed) or (max_len, n_embed)
        inputs_embeds = self.model.embed_tokens(input_ids) 
        
        if len(list(inputs_embeds.shape)) == 2:  # (max_len, n_embed)
            # (batch_size=1, max_len, n_embed) 
            inputs_embeds = inputs_embeds.unsqueeze(dim=0)

        if len(list(attention_mask.shape)) == 1:  # (max_len,)
            # (batch_size=1, max_len)
            attention_mask = attention_mask.unsqueeze(dim=0)

        # Get user_embeddings
        sep_indices = torch.argmax((input_ids == self.sep_id).to(torch.int), dim=1)
        # tokenized user_identifier are between the beginning bos token and the sep token
        user_input_ids = [input_ids[i, 1:sep_indices[i]] for i in range(input_ids.size(0))]
        # (batch_size, n_user_tokens, user_embed_dim)
        user_embeddings = self.user_model.get_user_embeddings(user_input_ids, generic_user=False) 

        if len(list(user_embeddings.shape)) == 1:  # (user_embed_dim, )
            user_embeddings = user_embeddings.unsqueeze(dim=0) # (batch_size=1, user_embed_dim)
        if len(list(user_embeddings.shape)) == 2:  # (batch_size, user_embed_dim)
            user_embeddings = user_embeddings.unsqueeze(dim=1) # (batch_size, n_user_tokens=1, user_embed_dim)

        new_input_embeds = []
        new_attention_masks = []

        for row_idx, row in enumerate(inputs_embeds):  # (batch_size, max_len, n_embed)
            user_embedding = user_embeddings[row_idx, :] if self.is_reference == False else \
                torch.full((0, user_embeddings[row_idx, :].size(1)), fill_value=0, dtype=row.dtype, 
                           device=user_embeddings.device)  # (n_user_tokens, user_embed_dim)
            pad_input_ids = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),),  
                                       fill_value=self.user_model.tokenizer.pad_token_id, 
                                       dtype=input_ids.dtype, 
                                       device=input_ids.device)
            pad_input_embeds = self.model.embed_tokens(pad_input_ids)
            # pad to the left so that the labels are valid when computing logps in DPO trainer
            # concatenate [padding for user_identifier, bos token, user_embedding, prompt + response + eos token + padding]
            new_input_embeds.append(torch.concat(
                [pad_input_embeds, row[:1, :], user_embedding, row[(sep_indices[row_idx]+1):, :]], 
                dim=0)
            )
            
            attention_mask_row = attention_mask[row_idx]  # (max_len,)
            user_attention_mask = torch.full(size=(user_embedding.size(0),), 
                                             fill_value=1, 
                                             device=attention_mask.device)  # (user_tokens,)
            pad_attention_mask = torch.full(size=(sep_indices[row_idx] - user_embedding.size(0),), 
                                            fill_value=0, 
                                            device=attention_mask.device)
            new_attention_masks.append(torch.concat(
                [pad_attention_mask, attention_mask_row[:1], user_attention_mask,
                 attention_mask_row[(sep_indices[row_idx]+1):]], 
                 dim=0)
            )
        
        # (batch_size, max_len, user_embed_dim)  
        inputs_embeds = torch.stack(new_input_embeds, dim=0).to(input_ids.device)
        attention_mask = torch.stack(new_attention_masks, dim=0).to(attention_mask.device)

        return inputs_embeds, attention_mask, user_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ): 
        # For DPO trainer, a separator is added between user identifier and the prompt, and then tokenized.
        inputs_embeds, attention_mask, user_embeddings = self._cat_user_embedding_to_input_sep(
            input_ids=input_ids, attention_mask=attention_mask)
        
        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels, 
            **kwargs,
        )

class UserLlamaForCausalLM(UserLlamaMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)