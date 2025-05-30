#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=meta-llama/Meta-Llama-3.1-8B-Instruct
max_length=50000
strategy=""
result_dir=llama31/k_2bit
n_bit=2

python -m tasks.eval_LongLive \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 128 \
            --max_length $max_length \
            --continue_gen True \
            --cache_implementation quantized \
            --cache_backend quanto \
            --cache_nbits $n_bit \
            --load_in_4bit true
