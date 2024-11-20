#!/bin/bash
main_process_port=10002   # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/generate.py \
    --n_users 1396 \
    --is_baseline 0 \
    --user_model cluster \
    --n_user_clusters 10 \
    --n_user_tokens 10 \
    --add_textual_info True \
    --lora_checkpoint your/lora/checkpoint/path \
    --selected_examples_path ./data/prism_selected_examples.json \
    --output_dir where/you/want/to/save/the/generated/json/file \
    --max_prompt_text_length 1400 \
    --generate_max_new_tokens 500 \
    --generate_min_new_tokens 32 \
    --seed 123 \
    --model_name your/base/llm/path \
    --model_class llama \
    --dataset prism \
    --initialize_from_vocab True \
    --sep "||" \