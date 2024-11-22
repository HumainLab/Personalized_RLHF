# P-RLHF: Personalized Language Modeling from Personalized Human Feedback 
This repository contains the code for our paper [Personalized Language Modeling from Personalized Human Feedback](https://arxiv.org/abs/2402.05133). We propose a general RLHF framework for fine-tuning LLMs using personalized preference data. In P-RLHF, we learn a separate user model in addition to the base LLM. Our implementation works with any existing perference optimization (*PO) algorithms.



## Step 0: Setup
To run the code, you need to install the following packages.
```
conda create -n prlhf python=3.11.9
conda activate prlhf
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install datasets==2.21.0
pip install transformers==4.44.2
pip install trl==0.10.1
pip install peft==0.12.0
pip install wandb==0.17.9
pip install pydantic==2.9.0
pip install pandas
```


## Step 1: Data Preparation

All data processing code for [TLDR](https://huggingface.co/datasets/openai/summarize_from_feedback), [PSOUPS](https://github.com/joeljang/RLPHF) and [PRISM](https://huggingface.co/datasets/HannahRoseKirk/prism-alignment) can be found in the `data/` folder. 


## Step 2: Fine-tune Personalized LLMs using P-RLHF
Implementation of P-RLHF can be found in the `prlhf/` folder. We provide implementation of the individual and cluster user model, but one can extend the user models by introducing new ones in `prlhf/user_model.py`.

To run `train_language_model_dpo.py`, below is a sample script. For more training scripts, check out the `scripts/` folder.

```
accelerate launch prlhf/train_language_model_dpo.py \
    --user_model $USER_MODEL_TYPE \
    --model_class $MODEL_CLASS \
    --model_name $BASE_LLM_PATH \
    --tokenizer_name $BASE_LLM_TOKENIZER_PATH \
    --dataset $DATASET_NAME
```


Note: If instead of P-DPO, one prefers to use other personalized preference optimization algorithms (P-*PO), one can change this by setting different `loss_type` for `UserDPOTrainer` since `UserDPOTrainer` inherents the `DPOTrainer` in the [TRL](https://github.com/huggingface/trl/) library.

## Step 3: Generate Personalized Responses
To generate responses using a personalized LLM, below is a sample script. For more generation-related scripts, check out the `scripts/` folder.

```
accelerate launch prlhf/generate.py \
    --user_model $USER_MODEL_TYPE \
    --lora_checkpoint $LORA_CKPT_PATH \
    --output_dir $GENERATION_OUTPUT_DIR \
    --model_name $BASE_LLM_PATH \
    --model_class $MODEL_CLASS \
    --dataset $DATASET_NAME 
```


## Step 4: Evaluate Personalized LLMs

To obtain the win-rate of the trained model on PSOUPS and PRISM, checkout the `evaluate/` folder.

## 🌟 Citation
Please cite the paper and star this repo if you use P-RLHF and find it interesting/useful, thanks! Open an issue if you have any questions.

```bibtex
@article{personalizedRLHF,
  title={Personalized language modeling from personalized human feedback},
  author={Li, Xinyu and Zhou, Ruiyang and Lipton, Zachary C and Leqi, Liu},
  journal={arXiv preprint arXiv:2402.05133},
  year={2024}
}
```