#!/bin/bash
main_process_port=10012   # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/generate.py \
    --n_users 6 \
    --is_baseline 0 \
    --user_model individual \
    --n_user_tokens 10 \
    --lora_checkpoint your/lora/checkpoint/path \
    --output_dir where/you/want/to/save/the/generated/json/file \
    --generate_max_new_tokens 1024 \
    --generate_min_new_tokens 32 \
    --seed 123 \
    --model_name your/base/llm/path \
    --model_class llama \
    --dataset psoups \
    --koala_prompts_path ./data/koala_eval_50.json \
    --initialize_from_vocab True \
    --sep "||" \
    --lora_alpha 32 \
    --lora_dropout 0.1 \