#!/bin/bash

# Upload S dataset (5.97B tokens)
huggingface-cli upload JamesConley/fineweb-sample-5.97B-512 data/llm_datasets/chinchilla/s/ . --repo-type dataset

# Upload M dataset (22.95B tokens)
huggingface-cli upload JamesConley/fineweb-sample-22.95B-512 data/llm_datasets/chinchilla/m/ . --repo-type dataset

# Upload L dataset (111.96B tokens)
huggingface-cli upload JamesConley/fineweb-sample-111.96B-512 data/llm_datasets/chinchilla/l/ . --repo-type dataset