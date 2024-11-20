#!/bin/bash
main_process_port=10023   # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/train_language_model_dpo.py \
    --is_baseline 1 \
    --user_model individual \
    --n_user_tokens 10 \
    --sanity_check False \
    --alpha 0 \
    --beta 0.5 \
    --seed 123 \
    --model_class gptj \
    --model_name your/base/llm/path \
    --tokenizer_name your/base/llm/tokenizer/path \
    --max_prompt_length 550 \
    --max_length 600 \
    --max_text_length 4800 \
    --dataset tldr \
    --user_file ./data/sup_users_top10.txt \
    --user_preference_file ./data/sup_users_preferences.txt \
    --downloads_data_path your/downloaded/openai_summarize_comparisons/file/path \
    --use_downloads True \
    --output_dir where/you/want/to/save/the/trained/lora/weights \
    --report_to wandb \
    --wandb_project gptj_dpo_base \
    --wandb_dir where/you/want/to/save/the/wandb/offline/logs \
    --initialize_from_vocab True \
    --sep "||" \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 150 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --logging_steps 10 \
    --save_steps 200 \
    --eval_steps 50 \
    