#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD

checkpoint=NingLab/eCeLLM-M
max_length=7500
result_dir=eCeLLM

python -m tasks.eval_LongLive \
            --pipeline LongLive \
            --result_dir $result_dir \
            --gen_model  $checkpoint \
            --gen_max_new_tokens 128 \
            --max_length $max_length \
