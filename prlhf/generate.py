import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
import torch
import json
import time

from peft import LoraConfig, get_peft_model
from transformers import (
    PreTrainedTokenizerFast, 
    HfArgumentParser, 
    set_seed,
    GenerationConfig,
)
from user_language_model import UserGPTNeoForCausalLM, UserGPTJForCausalLM, UserLlamaForCausalLM
from utils import build_tldr_prompts, build_psoups_prompts, build_prism_prompts


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for generation.
    """
    # lm parameters
    model_name: Optional[str] = field(
        default="EleutherAI/gpt-neo-125m",
        metadata={"help": "path to the base LM."})
    model_class: Optional[str] = field(
        default="gptneo",
        metadata={"help": "the model class, must be one of gptneo, gptj, or llama."})
    lora_checkpoint: Optional[str] = field(default=None, metadata={"help": "LoRA checkpoint path for generating outputs."})  
    
    # user model parameters
    is_baseline: Optional[int] = field(
        default=0, metadata={"help": "assuming running P-DPO if set to 0; vanilla DPO as baseline if set to 1."})
    user_model: Optional[str] = field(default="individual", metadata={"help": "user model type."})
    n_users: Optional[int] = field(default=10, metadata={"help": "number of users."})
    n_user_clusters: Optional[int] = field(default=5, metadata={"help": "number of user clusters."})
    n_user_tokens: Optional[int] = field(default=1, metadata={"help": "number of user tokens."})
    seed: Optional[int] = field(default=123)
    add_generic_user: Optional[bool] = field(
        default=True, metadata={"help": "whether to add generic user embedding to individual user embeddings."})
    initialize_from_vocab: Optional[bool] = field(
        default=True, metadata={"help": "whether to initialize user embeddings from vocabulary."})
    most_common_tokens: Optional[str] = field(
        default=None, metadata={"help": "torch file name including most commonly used tokens."})
    random_range: Optional[float] = field(default=0.5, metadata={"help": "random range to initialize user embeddings."})
    sep: Optional[str] = field(default="||", metadata={"help": "the separator between user identifier and text."}) 

    # peft parameters
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter."})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter."})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter."})

    # generation parameters
    dataset: Optional[str] = field(default="tldr", metadata={"help": "the dataset to use."})
    tldr_selected_prompts_path: Optional[str] = field(
        default=None, metadata={"help": "file path containing tldr prompts randomly selected for evaluation."})
    koala_prompts_path: Optional[str] = field(default=None, metadata={"help": "file path containing koala prompts for psoups evaluation."})
    selected_examples_path: Optional[str] = field(
        default=None, metadata={"help": "file path containing selected examples for prism evaluation."})
    add_textual_info: Optional[bool] = field(
        default=False, metadata={"help": "whether to add textual user information to the prompt for prism dataset."})
    max_prompt_text_length: Optional[int] = field(
        default=2400, metadata={"help": "max history text string length for prism, according to max_prompt_length."})
    generate_max_new_tokens: Optional[int] = field(default=1024, metadata={"help": "the maximum new tokens to generate."})
    generate_min_new_tokens: Optional[int] = field(default=1024, metadata={"help": "the minimum new tokens to generate."})
    output_dir: Optional[str] = field(default=None, metadata={"help": "the dir to save output responses json file."})


def main(script_args):
    # load the tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(script_args.lora_checkpoint)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "sep_token", None) is None:
        tokenizer.add_special_tokens({"sep_token": script_args.sep})

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
        n_users=script_args.n_users,
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

    # we could also add other llama modules into target_modules
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

    model = get_peft_model(model, peft_config)

    if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
        model.load_adapter(script_args.lora_checkpoint, model.active_adapter, is_trainable=False)
    print(f"Loaded LoRA weights from {script_args.lora_checkpoint}!")
   
    model.eval()
    model.cuda()

    # tokens covered by past_key_values will be omitted during generation if use_cache=True
    # which will cause trouble in getting user ids from tokenized user identifier
    # so set use_cache to False in GenerationConfig
    if script_args.dataset == "tldr" and script_args.model_class == "gptj":
        # for tldr dataset, evaluate using 50 prompts randomly sampled from the tldr validation set
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=0.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.15,
            max_new_tokens=script_args.generate_max_new_tokens,
            min_new_tokens=script_args.generate_min_new_tokens,
            use_cache=False,
        )
        prompts = build_tldr_prompts(script_args)

    if script_args.dataset == "psoups" and script_args.model_class == "llama":
        # for psoups dataset, evaluate using 50 prompts from koala
        generation_config = GenerationConfig(
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=0.0,
            top_p=1.0,
            no_repeat_ngram_size=3,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.15,
            max_new_tokens=script_args.generate_max_new_tokens,
            min_new_tokens=script_args.generate_min_new_tokens,
            use_cache=False,
        )
        prompts = build_psoups_prompts(script_args)

    # prism dataset
    if script_args.dataset == "prism" and script_args.model_class == "llama":
        # popular generation config for llama3
        generation_config = GenerationConfig(
            bos_token_id=128000,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=[128001, 128008, 128009],
            temperature=0.6, 
            top_k=0.0,
            top_p=0.9,
            no_repeat_ngram_size=3,
            do_sample=True,
            num_return_sequences=1,
            repetition_penalty=1.15,
            max_new_tokens=script_args.generate_max_new_tokens,
            min_new_tokens=script_args.generate_min_new_tokens,
            use_cache=False,
        )
        prompts = build_prism_prompts(script_args)
    
    if os.path.exists(script_args.output_dir) is False:
        os.makedirs(script_args.output_dir, exist_ok=True)
    filename = f"{script_args.lora_checkpoint.replace('/', '-')}.json"

    with torch.no_grad():
        for num, prompt_id in enumerate(list(prompts.keys())):
            start_time = time.time()
            print(f"Generating prompt num {num}; prompt_id {prompt_id}")
            # one by one
            input_text = prompts[prompt_id]["input_text"]
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            input_ids = input_ids.to('cuda')

            model_output = model.generate(
                input_ids=input_ids, 
                generation_config=generation_config,
            )
            output_n_tokens = len(model_output[0])
            model_output = tokenizer.batch_decode(model_output)

            # append output
            prompts[prompt_id]["output_text"] = model_output[0]
            prompts[prompt_id]["output_length"] = len(model_output[0]) - len(input_text)

            with open(os.path.join(script_args.output_dir, filename), "w") as f: 
                json.dump(prompts, f, indent=4)
            
            print(f"Time: {round(time.time()-start_time, 2)}, output n_tokens: {output_n_tokens}")
            print(f"Generated output:\n{model_output[0]}")


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    set_seed(script_args.seed)
    main(script_args)