import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import torch
from peft import LoraConfig
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed,
)
from trl import DPOConfig
from utils import load_openai_comparisons, load_psoups_comparisons, load_prism_comparisons
from user_language_model import UserGPTNeoForCausalLM, UserGPTJForCausalLM, UserLlamaForCausalLM
from user_dpo_trainer import UserDPOTrainer 


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """
    # LLM parameters
    model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "path to the base LLM."})
    model_class: Optional[str] = field(
        default="gptneo", metadata={"help": "the model class, must be one of gptneo, gptj or llama."})
    tokenizer_name: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "path to the tokenizer."})
    max_prompt_text_length: Optional[int] = field(
        default=2400, metadata={"help": "max history text string length for prism, according to max_prompt_length."})
    max_text_length: Optional[int] = field(default=4800, metadata={"help": "the maximum text length."})
    
    # user model parameters
    user_file: Optional[str] = field(default=None, metadata={"help": "path to file including selected original user IDs."})
    user_model: Optional[str] = field(
        default="individual", metadata={"help": "user model type, has to be either individual or cluster."})
    n_user_clusters: Optional[int] = field(
        default=5, metadata={"help": "number of user clusters if user_model is set to cluster."})
    n_user_tokens: Optional[int] = field(default=1, metadata={"help": "number of user tokens."})
    initialize_from_vocab: Optional[bool] = field(
        default=True, metadata={"help": "whether to initialize user embeddings from word embeddings in LLM vocabulary."})
    most_common_tokens: Optional[str] = field(
        default=None, metadata={"help": "path to a torch file which includes most commonly used tokens."})
    random_range: Optional[float] = field(default=0.5, metadata={"help": "random range to initialize user embeddings."})
    sep: Optional[str] = field(default="||", metadata={"help": "the separator between user identifier and prompt text."}) 
    seed: Optional[int] = field(default=123)
    add_generic_user: Optional[bool] = field(
        default=True, metadata={"help": "whether to add generic user embedding to individual user embeddings."})
    
    # DPO training/evaluation parameters
    beta: Optional[float] = field(default=0.5, metadata={"help": "the beta parameter for DPO loss."})
    alpha: Optional[float] = field(
        default=0.5, metadata={"help": "trade-off between individual user loss and generic user loss in P-DPO."})
    max_prompt_length: Optional[int] = field(
        default=512, metadata={"help": "the maximum tokenized prompt length for DPO trainer."})
    max_length: Optional[int] = field(
        default=1024, metadata={"help": "the maximum tokenized sequence length for DPO trainer."})
    learning_rate: Optional[float] = field(default=5e-5, metadata={"help": "DPO learning rate."})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type."})
    warmup_steps: Optional[int] = field(default=150, metadata={"help": "the number of warmup steps."})
    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device."})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device."})
    gradient_accumulation_steps: Optional[int] = field(
        default=8, metadata={"help": "the number of gradient accumulation steps."})
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing."})
    num_train_epochs: Optional[int] = field(default=2, metadata={"help": "the number of training epochs."})
    logging_steps: Optional[int] = field(default=10, metadata={"help": "the logging frequency."})
    save_steps: Optional[int] = field(default=100, metadata={"help": "the saving frequency."})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "the evaluation frequency."})

    # peft parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter."})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter."})

    # instrumentation
    dataset: Optional[str] = field(default="tldr", metadata={"help": "the dataset to use."})
    test_ratio: Optional[float] = field(default=0.1, metadata={"help": "the ratio of validation set for psoups dataset."})
    add_textual_info: Optional[bool] = field(
        default=False, metadata={"help": "whether to add textual user information to the prompt for prism dataset."})
    output_dir: Optional[str] = field(default="./dpo", metadata={"help": "the output directory."})
    output_postfix: Optional[str] = field(default=None, metadata={"help": "postfix to the run name."})
    sanity_check: Optional[bool] = field(
        default=False, metadata={"help": "only train on 1000 samples and evaluate on 500 samples for sanity check."})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    wandb_project: Optional[str] = field(default="dpo", metadata={"help": "project name on WANDB."})
    wandb_dir: Optional[str] = field(default="./wandb", metadata={"help": "directory to save WANDB log files."})
    use_downloads: Optional[bool] = field(default=False, metadata={"help": "use cached TL;DR data if set to True."})
    downloads_data_path: Optional[str] = field(default="./data", metadata={"help": "the path to saved dataset."})
    is_baseline: Optional[int] = field(
        default=0, metadata={"help": "assuming running P-DPO if set to 0; vanilla DPO as baseline if set to 1."})
    user_preference_file: Optional[str] = field(
        default=None, metadata={
            "help": "path to file which includes the synthetic user preferences for TL;DR dataset,"
            "0 means preferring longer responses while 1 means preferring shorter ones."})
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "whether to resume from the latest checkpoint."})
    resume_output_dir: Optional[str] = field(
        default=None, metadata={"help": "the output directory to resume from."})
    

def main(script_args):
    # output sub-directory name
    output_path = f"{script_args.model_class}_b{script_args.beta}_a{script_args.alpha}_"\
        f"lr{script_args.learning_rate}_wm{script_args.warmup_steps}_lr{script_args.lr_scheduler_type}_seed{script_args.seed}"
    
    if script_args.is_baseline == 1:
        output_path += "_base"
    else:
        if script_args.user_model == "cluster":
            output_path += f"_uc{script_args.n_user_clusters}"
        elif script_args.user_model == "individual":
            output_path += f"_ind"
    
    output_path += f"_ut{script_args.n_user_tokens}"

    if script_args.initialize_from_vocab:
        output_path += "_vcb"
    else:
        output_path += "_rdm"

    if script_args.dataset == "prism" and script_args.add_textual_info:
        output_path += "_textual"
    if script_args.output_postfix is not None:
        output_path += f"_{script_args.output_postfix}"

    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": script_args.sep})

    # load the comparison datasets     
    if script_args.dataset == "tldr":
        train_dataset, eval_dataset, n_users = load_openai_comparisons(
            user_file=script_args.user_file, 
            sep=script_args.sep, 
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length, 
            sanity_check=script_args.sanity_check,
            use_downloads=script_args.use_downloads,
            downloads_data_path=script_args.downloads_data_path,
            user_preference_file=script_args.user_preference_file,
        )
    elif script_args.dataset == "psoups":
        train_dataset, eval_dataset, n_users = load_psoups_comparisons(
            sep=script_args.sep, 
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            test_ratio=script_args.test_ratio,
            sanity_check=script_args.sanity_check,
            downloads_data_path=script_args.downloads_data_path,
            seed=script_args.seed,
        )

    elif script_args.dataset == "prism":
        train_dataset, eval_dataset, n_users = load_prism_comparisons(
            sep=script_args.sep, 
            n_user_tokens=script_args.n_user_tokens,
            max_text_length=script_args.max_text_length,
            sanity_check=script_args.sanity_check,
            seed=script_args.seed,
            prism_data_path=script_args.downloads_data_path,
            max_prompt_string_length=script_args.max_prompt_text_length,
            add_textual_info=script_args.add_textual_info,
        )

    print(f"Loaded {script_args.dataset} dataset, train: {len(train_dataset)}, " 
          f"eval: {len(eval_dataset)} n_users: {n_users}")

    # most common tokens used to initialize user embeddings
    most_common_tokens = torch.load(script_args.most_common_tokens) \
        if script_args.most_common_tokens is not None else None

    # load pretrained models
    if script_args.model_class == "gptneo":
        model_class = UserGPTNeoForCausalLM
    elif script_args.model_class == "gptj":
        model_class = UserGPTJForCausalLM
    elif script_args.model_class == "llama":
        model_class = UserLlamaForCausalLM

    model = model_class.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name,
        user_model_type=script_args.user_model,
        tokenizer=tokenizer,
        n_users=n_users,
        n_clusters=script_args.n_user_clusters,
        n_user_tokens=script_args.n_user_tokens,
        seed=script_args.seed,
        add_generic_user=script_args.add_generic_user,
        initialize_from_vocab=script_args.initialize_from_vocab,
        most_common_tokens=most_common_tokens,
        random_range=script_args.random_range,
        sep=script_args.sep,
        is_reference=False if script_args.is_baseline == 0 else True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32,
    )
    model.config.use_cache = False

    ref_model = model_class.from_pretrained(
        pretrained_model_name_or_path=script_args.model_name,
        user_model_type=script_args.user_model,
        tokenizer=tokenizer,
        n_users=n_users,
        n_clusters=script_args.n_user_clusters,
        n_user_tokens=script_args.n_user_tokens,
        seed=script_args.seed,
        add_generic_user=script_args.add_generic_user,
        initialize_from_vocab=script_args.initialize_from_vocab,
        most_common_tokens=most_common_tokens,
        random_range=script_args.random_range,
        sep=script_args.sep,
        is_reference=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    
    # initialize training arguments
    if script_args.report_to == "wandb" and script_args.wandb_project is not None:
        if not os.path.exists(script_args.wandb_dir):
            os.makedirs(script_args.wandb_dir, exist_ok=True)
        os.environ["WANDB_PROJECT"] = script_args.wandb_project
        os.environ['WANDB_DIR'] = script_args.wandb_dir

    training_args = DPOConfig(
        beta=script_args.beta,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        truncation_mode="keep_start",
        force_use_ref_model=True,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        eval_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=os.path.join(script_args.output_dir, output_path) if script_args.resume_output_dir is None else script_args.resume_output_dir,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_steps=script_args.warmup_steps,
        bf16=True,
        remove_unused_columns=False,
        report_to=script_args.report_to,
        run_name=output_path,
    )

    # we could also add other llama modules into peft target_modules
    # e.g. ["o_proj", "up_proj", "down_proj", "gate_proj", "embed_tokens",] 
    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        modules_to_save=["user_embedding", "cluster_embedding", "user_weight"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # initialize the personalized DPO trainer
    pdpo_trainer = UserDPOTrainer(
        alpha=script_args.alpha,
        sep=script_args.sep,
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
    )

    pdpo_trainer.train(resume_from_checkpoint=True if script_args.resume_from_checkpoint else None)
    final_ckpt_path = os.path.join(script_args.output_dir, output_path, "final_ckpt") \
        if script_args.resume_output_dir is None else os.path.join(script_args.resume_output_dir, "final_ckpt")
    pdpo_trainer.save_model(final_ckpt_path)


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    main(script_args)