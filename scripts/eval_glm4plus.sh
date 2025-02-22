#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=""
max_length=50000
strategy=""
result_dir=glm4plus
n_bit=4

python -m tasks.eval_LongLive_api \
            --pipeline LongLive \
            --result_dir $result_dir \