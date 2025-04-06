#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
PWD="$(pwd)" 
cd $PWD


result_dir=gpt4o4/
models=gpt4o.yaml

python -m tasks.eval_LongLive_api \
            --pipeline LongLive \
            --result_dir $result_dir \
            --models $models \
            # --continue_gen True \

