#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=NingLab/eCeLLM-M
max_length=50000
strategy=""
result_dir=eCeLLM
n_bit=4

python -m tasks.eval_LongLive \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 128 \
            --max_length $max_length \
            # --cache_implementation quantized \
            # --cache_backend quanto \
            # --cache_nbits $n_bit \
            # --load_in_4bit true
