# Win-rate evaluation

In order to keep track of the randomization of input orders, we implement it in the function `to_annotate_format()` in `evaluate/winrate_gpt_evaluation.ipynb`. Therefore, we disable the randomization in alpaca-farm by changing Line 570 in `alpaca_farm/auto_annotations/pairwise_annotators.py` to `is_randomize_output_order: bool = False`.