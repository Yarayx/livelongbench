#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=Qwen/Qwen2.5-VL-7B-Instruct
max_length=7500
strategy=""
result_dir=Qwen2.5
n_bit=4

python -m tasks.eval_LongLive \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 128 \
            --max_length $max_length \
            --continue_gen True \

