#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD


checkpoint=meta-llama/Meta-Llama-3.1-8B-Instruct
max_length=50000
strategy=""
result_dir=llama31/l_2x_0213
n_bit=4

python -m tasks.eval_LongLive_lingua \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 128 \
            --max_length $max_length \
            --continue_gen True \
            # --cache_implementation quantized \
            # --cache_backend quanto \
            # --cache_nbits $n_bit \
            # --load_in_4bit true
