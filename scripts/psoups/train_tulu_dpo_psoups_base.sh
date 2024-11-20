#!/bin/bash
main_process_port=10013   # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/train_language_model_dpo.py \
    --is_baseline 1 \
    --user_model individual \
    --n_user_tokens 10 \
    --sanity_check False \
    --alpha 0 \
    --beta 0.1 \
    --seed 123 \
    --model_class llama \
    --model_name your/base/llm/path \
    --tokenizer_name your/base/llm/tokenizer/path \
    --max_prompt_length 200 \
    --max_length 1000 \
    --max_text_length 4800 \
    --dataset psoups \
    --test_ratio 0.1 \
    --downloads_data_path ./data/allcombo_8_cleaned.json \
    --use_downloads True \
    --output_dir where/you/want/to/save/the/trained/lora/weights \
    --report_to wandb \
    --wandb_project tulu_dpo_base \
    --wandb_dir where/you/want/to/save/the/wandb/offline/logs \
    --initialize_from_vocab True \
    --sep "||" \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 150 \
    --lora_alpha 32 \
    --lora_dropout 0.1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 50 \
    