#!/bin/bash

# Create S dataset
python3 pull_tokens.py --input_folder data/llm_datasets/fineweb-edu-350BT/sample/350BT/ --output_file data/llm_datasets/chinchilla/s/train.parquet --num_tokens 5968954880 --seed 42

# Create M dataset
python3 pull_tokens.py --input_folder data/llm_datasets/fineweb-edu-350BT/sample/350BT/ --output_file data/llm_datasets/chinchilla/m/train.parquet --num_tokens 22948244480 --seed 42

# Create L dataset
python3 pull_tokens.py --input_folder data/llm_datasets/fineweb-edu-350BT/sample/350BT/ --output_file data/llm_datasets/chinchilla/l/train.parquet --num_tokens 111957923840 --seed 42