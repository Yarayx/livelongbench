import logging
logger = logging.getLogger("main")

import os.path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM
from src.models import init_args, HuggingFaceModel
from accelerate import Accelerator
from transformers import HfArgumentParser
from src import ModelArgs
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List

@dataclass
class Args(ModelArgs):
    exp_desc: str = field(
        default="",
        metadata={'help': 'Experiment description.'}
    )
    pipeline_config_dir: str = field(
        default="",
        metadata={'help': 'Directory for pipeline configuration.'}
    )
    eval_config_dir: str = field(
        default="",
        metadata={'help': 'Directory for evaluation configuration.'}
    )
    output_folder_dir: str = field(
        default="",
        metadata={'help': 'Directory for output results.'}
    )
    eval_data: str = field(
        default="../data/longbench/test.json",
        metadata={'help': 'The evaluation json data path.'}
    )
    output_dir: str = field(
        default="../data/results/longbench/",
        metadata={'help': 'The base directory for saving results and logs.'}
    )
    result_dir: Optional[str] = field(
        default=None,
        metadata={'help': 'The directory relative to output_dir for saving results.'}
    )

    dataset_names: List[str] = field(
        default_factory=lambda: ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique', 'gov_report', 'qmsum', 'multi_news'],
        metadata={'help': 'Which dataset to evaluate?'}
    )

    max_length: Optional[int] = field(
        default=None,
        metadata={'help': 'Max input length.'}
    )
    truncate_from_middle: bool = field(
        default=True,
        metadata={'help': 'Truncate inputs from the middle.'}
    )
    load_result: bool = field(
        default=False,
        metadata={'help': 'Load result from saved files?'}
    )
    mini: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for each dataset?'}
    )
    

def initialize_model_tokenizer(pipeline_params):
    config = AutoConfig.from_pretrained(pipeline_params['model_name'])
    parser = HfArgumentParser([Args])
    args_dict = {
        'cache_implementation': pipeline_params.get('cache_implementation'),
        'cache_backend': pipeline_params.get('cache_backend'),
        'cache_nbits': pipeline_params.get('cache_nbits'),
        'load_in_4bit': pipeline_params.get('load_in_4bit', False),
    }
    args, = parser.parse_dict(args_dict)  # Unpack single item tuple
    accelerator = Accelerator(cpu=args.cpu)
    model_args=args
    device=accelerator.device

    model_kwargs, tokenizer_kwargs = init_args(
        model_args, 
        model_args.gen_model, device)
    
    if 'rope_theta_factor' in pipeline_params and hasattr(config, 'rope_theta'):
        config.rope_theta *= pipeline_params['rope_theta_factor']

    if pipeline_params['use_flash_attn']:
        attn_implementation = 'flash_attention_2'
    else:
        attn_implementation = 'eager'

    model = HuggingFaceModel(
                pipeline_params['model_name'],
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )

    tokenizer = AutoTokenizer.from_pretrained(pipeline_params['tokenizer_name'], padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    logger.info(f'Model {model} and Tokenizer {tokenizer} initialized.')

    return model, tokenizer

def prompt_compressor(compressor, original_prompt, rate):
    cp_pt_list = []
    cp_pt_len_list = []
    for prompt in tqdm(original_prompt):
        results = compressor.compress_prompt_llmlingua2(
            prompt,
            rate=rate,
            force_tokens=['\n', '.', '!', '?', ','],
            chunk_end_tokens=['.', '\n'],
            return_word_label=True,
            drop_consecutive=True
        )
        cp_pt_list.append(results['compressed_prompt'])
        cp_pt_len_list.append(results['compressed_tokens'])

    return cp_pt_list, cp_pt_len_list

def batch_generate(batched_input, model,tokenizer,max_new_tokens, **kwargs):

    if isinstance(batched_input[0], str):
        model_inputs = tokenizer(batched_input, add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
        input_length = model_inputs.input_ids.shape[1]
        inputs = model_inputs.input_ids
    elif isinstance(batched_input, torch.Tensor):
        inputs = batched_input.to("cuda")
        input_length = batched_input.shape[1]
    else:
        logger.error(f"Unknown batched_input:{batched_input}")
        raise ValueError

    generated_ids = model.generate(inputs, do_sample=False, max_new_tokens=max_new_tokens, **kwargs)
    responses = tokenizer.batch_decode(generated_ids[:, input_length:], skip_special_tokens=True)

    return responses
