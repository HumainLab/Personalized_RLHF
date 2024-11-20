import os
import numpy as np
from datasets import load_dataset, load_from_disk, Dataset
import json


def encode_selected_users(file_name="./selected_users.txt"):
    user_list = np.genfromtxt(file_name, dtype="str")
    selected_users = {}
    for i, user in enumerate(user_list):
        # seen workers are numbered as 1, 2, ...
        # worker 0 is reserved for the generic worker
        selected_users[user] = i+1  
    return selected_users 


def build_tldr_dataset_dpo_synthetic(
    dataset,
    selected_users,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    user_preference_file="./users_preference.txt",
    original_columns=None,
    num_proc=24,
):
    '''Build synthetic dataset for DPO from `openai/summarize_from_feedback`.
       The preferences of users depend on the length of the responses.
    '''
    
    user_list = np.genfromtxt(user_preference_file, dtype="str")
    user_preference = {}
    for i in range(user_list.shape[0]):
        user_preference[user_list[i, 0]] = int(user_list[i, 1])

    def preprocess_function(dataset):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }
        for info, summaries, choice, worker in \
            zip(dataset["info"], dataset["summaries"], dataset["choice"], dataset["worker"]):
            original_text_field = "post" if info["post"] is not None else "article"
            query = f"SUBREDDIT: r/{info['subreddit']}\nTITLE: {info['title']}\nPOST: {info[original_text_field]}\nTL;DR: "
            if summaries[0]["text"] == summaries[1]["text"]:
                continue

            len_0, len_1 = len(summaries[0]["text"]), len(summaries[1]["text"])
            # if preference = 0, user prefers longer summary
            # if preference = 1, user prefers shorter summary
            if ((len_0 >= len_1) and (user_preference[worker] == 0)) or \
                ((len_0 <= len_1) and (user_preference[worker] == 1)):
                summary_j = summaries[0]["text"]
                summary_k = summaries[1]["text"]
            
            if ((len_0 < len_1) and (user_preference[worker] == 0)) or \
                ((len_0 > len_1) and (user_preference[worker] == 1)):
                summary_j = summaries[1]["text"]
                summary_k = summaries[0]["text"]

            if selected_users is not None:
                if worker in selected_users.keys():
                    user_id = selected_users[worker]
                else:
                    user_id = 0
            else:
                user_id = 0

            # for personalized dpo, prompt = user identifier|query
            user_identifier = f"USER: {user_id}" + ("<|endoftext|>"*n_user_tokens)
            prompt = user_identifier + sep + query

            # is_train = True: use selected users in train split for training
            # is_train = False: use all users in validation split for evaluation
            if (is_train and (user_id != 0)) or (not is_train):
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(summary_j)
                new_examples["rejected"].append(summary_k)

            if (is_train and (user_id != 0)):
                # for each example used for training (user_id != 0),
                # create a duplicate example with user_id = 0 to simplify the loss computation
                user_identifier = f"USER: 0" + ("<|endoftext|>"*n_user_tokens)
                prompt = user_identifier + sep + query
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(summary_j)
                new_examples["rejected"].append(summary_k)
            
        return new_examples 

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    print("Finished loading synthetic TL;DR dataset!")
    return ds


def load_openai_comparisons(
    user_file="./selected_users.txt",
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    sanity_check=False,
    use_downloads=False,
    downloads_data_path=None,
    user_preference_file="./users_preference.txt",
    num_proc=24,
):
    # the selected users
    selected_users = encode_selected_users(file_name=user_file) if user_file is not None else None
    n_users = len(selected_users)

    if not use_downloads:
        train_dataset = load_dataset("openai/summarize_from_feedback", name="comparisons", split="train")
        eval_dataset  = load_dataset("openai/summarize_from_feedback", name="comparisons", split="validation")
    else:
        # load data from the path which contains downloaded data
        assert downloads_data_path is not None, "downloads_data_path cannot be None if use_downloads is True!"
        train_dataset = load_from_disk(os.path.join(downloads_data_path, "train"))
        eval_dataset  = load_from_disk(os.path.join(downloads_data_path, "validation"))

    # only use comparisons sampled from SFTs (but not ppo policies)
    filter_func = lambda x: all(y["policy"].find("sup") >= 0 and y["policy"].find("ppo") == -1 and \
                                y["policy"].find("cnn") == -1 for y in x["summaries"])
    train_dataset = train_dataset.filter(filter_func)
    eval_dataset  = eval_dataset.filter(filter_func)

    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(100))
        eval_dataset  = eval_dataset.select(range(50))
    original_columns = train_dataset.column_names

    datasets = {"train": train_dataset, "eval": eval_dataset}
    for key in datasets.keys():
        assert user_preference_file is not None, \
            "user_preference_file cannot be None if is_synthetic is True!"
        datasets[key] = build_tldr_dataset_dpo_synthetic(
            dataset=datasets[key],
            selected_users=selected_users,
            sep=sep, 
            n_user_tokens=n_user_tokens, 
            is_train=True if key=="train" else False,
            user_preference_file=user_preference_file,
            original_columns=original_columns,
            num_proc=num_proc,
        )
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )

    return datasets["train"], datasets["eval"], n_users


def build_psoups_dataset_dpo(
    dataset,
    sep="||", 
    n_user_tokens=1,
    is_train=True,
    original_columns=None,
    num_proc=24,
):
    '''Build dataset for DPO from personalized soups allcombo_8_cleaned.json generated by running /data/psoups_dataset.ipynb.'''
    
    def preprocess_function(dataset):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        for user_id, question, response_j, response_k in \
            zip(dataset["user_id"], dataset["user_input"], dataset["completion_a"], dataset["completion_b"]):
            question = f"<|user|>\n{question} \n<|assistant|>\n"
            # for personalized dpo, prompt = user identifier|query
            user_identifier = f"USER: {user_id} " + ("</s> "*n_user_tokens)
            prompt = user_identifier + sep + ' ' + question

            if (is_train and (user_id != 0)) or (not is_train):
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(response_j)
                new_examples["rejected"].append(response_k)
                    
            if (is_train and (user_id != 0)):
                # for each example used for training (user_id != 0),
                # create a duplicate example with user_id = 0 to simply the loss computation
                user_identifier = f"USER: 0 " + ("</s> "*n_user_tokens)
                prompt = user_identifier + sep + ' ' + question
                new_examples["prompt"].append(prompt)
                new_examples["chosen"].append(response_j)
                new_examples["rejected"].append(response_k)
        
        return new_examples    

    ds = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )
    print("Finished loading psoups dataset!")
    return ds   


def load_psoups_comparisons(
    sep="||",
    n_user_tokens=1,
    max_text_length=4800,
    test_ratio=0.1,
    sanity_check=False,
    downloads_data_path=None,
    num_proc=24,
    seed=123,
):
    # the train split was automatically created by the load_dataset function
    # when loading a json file
    dataset = load_dataset("json", data_files=downloads_data_path, split="train")
    dataset = dataset.shuffle(seed=seed)
    # split into train and validation datasets
    dataset = dataset.train_test_split(test_size=test_ratio, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    if sanity_check: # use a small subset for sanity check
        train_dataset = train_dataset.select(range(100))
        eval_dataset  = eval_dataset.select(range(50))
    original_columns = train_dataset.column_names

    datasets = {"train": train_dataset, "eval": eval_dataset}
    for key in datasets.keys():
        datasets[key] = build_psoups_dataset_dpo(
            dataset=datasets[key],
            sep=sep, 
            n_user_tokens=n_user_tokens,
            is_train=True if key=="train" else False,
            original_columns=original_columns,
            num_proc=num_proc,
        )
    
        datasets[key] = datasets[key].filter(
            lambda x: len(x["prompt"]) + len(x["chosen"]) <= max_text_length \
                and len(x["prompt"]) + len(x["rejected"]) <= max_text_length
        )

    return datasets["train"], datasets["eval"], 6  # n_users = 6 for psoups


def load_prism_comparisons(
    sep="||",
    n_user_tokens=10,
    max_text_length=2400,
    max_prompt_string_length = 1500,  # about 500 tokens, corresponds to max_prompt_length 550
    sanity_check=False,
    prism_data_path=None,
    seed=123,
    add_textual_info=False,
):
    with open(os.path.join(prism_data_path, "prism_data_dialog.json"), 'r') as f:
        data_dialog = json.load(f)
    with open(os.path.join(prism_data_path, "prism_data_user.json"), 'r') as f:
        data_user = json.load(f)
    with open(os.path.join(prism_data_path, "prism_split_ids.json"), 'r') as f:
        split_ids = json.load(f)

    n_users = len(data_user)

    def preprocess_function(is_train, add_textual_info=False):
        new_examples = {
            "prompt": [],
            "chosen": [],
            "rejected": [],
        }

        if is_train:
            dialog_ids = split_ids["train_dialog_ids"]
            user_ids = split_ids["seen_user_ids"]
        else:
            dialog_ids = split_ids["test_dialog_ids"]
            user_ids = split_ids["seen_user_ids"]
            user_ids.update(split_ids["unseen_user_ids"])

        for dialog_id in dialog_ids:
            history = ""
            # textual info of the user
            if add_textual_info:
                preference = ", ".join(data_user[data_dialog[dialog_id]["user_id"]]["demographics"]["preference"])
                textual_info = f"preference: {preference}; "
                for k in data_user[data_dialog[dialog_id]["user_id"]]["demographics"]:
                    if k != "preference":
                        textual_info += f"{k}: {data_user[data_dialog[dialog_id]['user_id']]['demographics'][k]}; "
            for turn in data_dialog[dialog_id]["turns"]:
                # add user utterance to history
                history += f"<|start_header_id|>user<|end_header_id|>\n\n{turn['user_utterance'][0]}<|eot_id|>\n"

                # prepare examples, skip empty or too long examples
                if (    turn['user_utterance'] != [] 
                    and turn['chosen_utterance'] != [] 
                    and turn['rejected_utterance'] != [] 
                    and len(turn["user_utterance"][0]) + len(turn["chosen_utterance"][0]) < max_text_length
                ):
                    # build user identifier
                    user_identifier = f"USER: {user_ids[data_dialog[dialog_id]['user_id']]} " + ("<|end_of_text|>"*n_user_tokens)
                        
                    # build prompt
                    prompt = user_identifier + sep
                    if add_textual_info:
                        prompt += 'User textual information: ' + textual_info 
                    # truncate history
                    max_history_string_length = max_prompt_string_length - len(prompt)
                    if len(history) > max_history_string_length:
                        history = history[-max_history_string_length:]
                    prompt += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"

                    # append to examples
                    for rejected in turn["rejected_utterance"]:
                        # skip too long examples
                        if len(turn["user_utterance"][0]) + len(rejected) > max_text_length:
                            continue
                        new_examples["prompt"].append(prompt)
                        new_examples["chosen"].append(turn['chosen_utterance'][0])
                        new_examples["rejected"].append(rejected)
                        # if train, duplicate 0
                        if is_train:
                            user_identifier_0 = f"USER: {0} " + ("<|end_of_text|>"*n_user_tokens)
                            prompt_0 = user_identifier_0 + sep
                            if add_textual_info:
                                user_identifier_0 += 'User textual information: ' + textual_info 
                            prompt_0 += '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
                            new_examples["prompt"].append(prompt_0)
                            new_examples["chosen"].append(turn['chosen_utterance'][0]) 
                            new_examples["rejected"].append(rejected)
                    
                # add the first chosen utterance to history for next turn
                if turn['chosen_utterance'] != []:
                    history += f"<|start_header_id|>assistant<|end_header_id|>\n\n{turn['chosen_utterance'][0]}<|eot_id|>\n"

        return new_examples

    train_examples = preprocess_function(is_train=True, add_textual_info=add_textual_info)
    test_examples = preprocess_function(is_train=False, add_textual_info=add_textual_info)
    
    train_dataset = Dataset.from_dict(train_examples)
    test_dataset = Dataset.from_dict(test_examples)

    # use a small subset for sanity check
    if sanity_check: 
        train_dataset = train_dataset.select(range(100))
        test_dataset  = test_dataset.select(range(50))
    
    return train_dataset, test_dataset, n_users


def build_tldr_prompts(script_args):
    tldr_selected_prompts = np.load(script_args.tldr_selected_prompts_path)

    if script_args.is_baseline == 1:
        # for vanilla DPO baseline, only need to generate once 
        user_ids = [0]
    else:
        # for known users seen during training, user_id = 1, ..., n_users
        # for unknown users, user_id = 0
        user_ids = range(0, script_args.n_users + 1) 

    tldr_prompts = {} 
    for prompt_id, prompt in enumerate(tldr_selected_prompts):
        for user_id in user_ids:
            # DPO trainer adds a bos token to the beginning of the input text
            # since GPT-J tokenizer does not add bos token automatically, we add it manually to the beginning
            user_identifier = "<|endoftext|>" + f"USER: {user_id}" + ("<|endoftext|>"*script_args.n_user_tokens)
            input_text = user_identifier + script_args.sep + prompt
            tldr_prompt = {
                "prompt_id": f"{user_id}_{prompt_id}",
                "our_user_id": user_id,
                "prompt": prompt,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            tldr_prompts[tldr_prompt["prompt_id"]] = tldr_prompt

    return tldr_prompts


def build_psoups_prompts(script_args):
    with open(script_args.koala_prompts_path, "r") as f:
        koala_prompts = json.load(f)
    
    if script_args.is_baseline == 1:
        # for vanilla DPO baseline, only need to generate once 
        user_ids = [0]
    else:
        # for known users seen during training, user_id = 1, ..., n_users
        # for unknown users, user_id = 0
        user_ids = range(0, script_args.n_users + 1) 

    psoups_prompts = {} 
    for prompt_id, prompt in enumerate(koala_prompts):
        prompt = prompt["prompt"]
        prompt = f"<|user|>\n{prompt} \n<|assistant|>\n"

        for user_id in user_ids:
            user_identifier = f"USER: {user_id} " + ("</s> "*script_args.n_user_tokens)
            input_text = user_identifier + script_args.sep + ' ' + prompt
            psoups_prompt = {
                "prompt_id": f"{user_id}_{prompt_id}",
                "our_user_id": user_id,
                "prompt": prompt,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            psoups_prompts[psoups_prompt["prompt_id"]] = psoups_prompt
    
    return psoups_prompts


def build_prism_prompts(script_args):
    with open(script_args.selected_examples_path, 'r') as f:
        selected_examples = json.load(f)

    prism_prompts = {}

    for dialog_id, dialog in selected_examples.items():
        # build user identifier
        user_identifier = f"USER: {dialog['our_id']} " + ("<|end_of_text|>"*script_args.n_user_tokens)
        for turn_nb, turn in dialog["turns"].items():
            # add textual user info
            if script_args.add_textual_info:
                preference = ", ".join(dialog["demographics"]["preference"])
                textual_info = f"preference: {preference}; "
                for k in dialog["demographics"]:
                    if k != "preference":
                        textual_info += f"{k}: {dialog['demographics'][k]}; "
            # build prompt
            prompt = user_identifier + script_args.sep
            if script_args.add_textual_info:
                prompt += 'User textual information: ' + textual_info 
            # truncate history
            max_history_string_length = script_args.max_prompt_text_length - len(prompt)
            if len(turn["history"]) > max_history_string_length:
                history = turn["history"][-max_history_string_length:]
            else:
                history = turn["history"]
            # append history to input
            input_text = prompt + '\n' + history + "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            prism_prompt = {
                "prompt_id": f"{dialog['user_id']}_{dialog_id}_{turn['turn_nb']}",
                "our_user_id": dialog['our_id'],
                "prompt": prompt,
                "truncated_history": history,
                "input_text": input_text,
                "output_text": "",
                "output_length": 0,
            }
            prism_prompts[prism_prompt["prompt_id"]] = prism_prompt

    return prism_prompts