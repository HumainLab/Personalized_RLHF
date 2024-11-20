#!/bin/bash
main_process_port=10005   # normally a random num > 10000
export WANDB_MODE=offline

accelerate launch --multi_gpu --num_processes=2 --main_process_port=${main_process_port} ./prlhf/train_language_model_dpo.py \
    --is_baseline 0 \
    --user_model cluster \
    --n_user_clusters 10 \
    --n_user_tokens 10 \
    --sanity_check False \
    --alpha 0.5 \
    --beta 0.5 \
    --seed 123 \
    --model_name your/base/llm/path \
    --model_class llama \
    --tokenizer_name your/base/llm/tokenizer/path \
    --max_prompt_length 470 \
    --max_prompt_text_length 1400 \
    --max_length 770 \
    --max_text_length 2300 \
    --dataset prism \
    --downloads_data_path ./data \
    --use_downloads False \
    --output_dir where/you/want/to/save/the/trained/lora/weights \
    --report_to wandb \
    --wandb_project lalma_dpo_cluster \
    --wandb_dir where/you/want/to/save/the/wandb/offline/logs \
    --add_textual_info True \
    --initialize_from_vocab True \
    --sep "||" \
    --learning_rate 5e-5 \
    --lr_scheduler_type cosine \
    --warmup_steps 150 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 1 \
    --logging_steps 10 \
    --save_steps 250 \
    --eval_steps 50 \