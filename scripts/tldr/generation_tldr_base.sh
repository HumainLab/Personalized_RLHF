#!/bin/bash
main_process_port=10021  # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/generate.py \
    --is_baseline 1 \
    --user_model individual \
    --n_users 10 \
    --n_user_tokens 10 \
    --lora_checkpoint your/lora/checkpoint/path \
    --output_dir where/you/want/to/save/the/generated/json/file \
    --generate_max_new_tokens 1024 \
    --generate_min_new_tokens 32 \
    --seed 123 \
    --model_name your/base/llm/path \
    --model_class gptj \
    --dataset tldr \
    --tldr_selected_prompts_path ./data/tldr_selected_prompts.npy \
    --initialize_from_vocab True \
    --sep "||" \