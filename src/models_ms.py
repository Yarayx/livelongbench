from typing import Dict, Union, List, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils import logging
import torch
from transformers.integrations import is_deepspeed_zero3_enabled
from tqdm import tqdm

logger = logging.get_logger(__name__)
from src import SelfExtend
from minference import MInference
from semantic_text_splitter import TextSplitter
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TokenizerWrapper:
    def __init__(self, tokenizer: PreTrainedTokenizerFast):
        self.tokenizer = tokenizer

    def __call__(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def to_str(self, token_ids):
        return self.tokenizer.decode(token_ids)

class HuggingFaceModel:
    def __init__(self, model_name_or_path, model_kwargs:Dict={}, tokenizer_kwargs:Dict={}):
        
        # NOTE: very important to add eval(), especially for ultragist models
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            **tokenizer_kwargs,
        )
        
        # model = Model(model_name_or_path)
        # print("model_name_or_path:",model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, 
            **model_kwargs,
        ).eval()

        # max_tokens = self.model.config.max_position_embeddings
        # wrapped_tokenizer = TokenizerWrapper(self.tokenizer)
        # self.splitter = TextSplitter.from_callback(wrapped_tokenizer, max_tokens)

        print(model_name_or_path)
        model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        minference_patch = MInference("minference", model_name) # inf_llm # minference #streaming2
        self.model=minference_patch(self.model)
        SelfExtend.apply(self.model, 32, 2048, enable_flash_attention=True, flash_attention_impl="flash_attn")

        self.model_name_or_path = model_name_or_path
        # use eos as pad
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # logger.info(f"Model config: {self.model.config}")

    def template2ids(self, template, remove_symbol=None):
        to_encode = self.tokenizer.apply_chat_template(
                template, 
                tokenize=False, 
                add_generation_prompt=True)

        if remove_symbol:
            to_encode = to_encode.replace(remove_symbol, "")

        inputs = self.tokenizer(
                to_encode, add_special_tokens=False, return_tensors="pt", padding=True
                ).to(self.model.device)

        return inputs


    def generate_conv(self, query, context, prompt, instruct:Union[str,list], **generation_kwargs):
        if isinstance(instruct, str):
            instruct = [instruct]
        context = [
            {"role": "user", "content": prompt.format(context=context)},
            {"role": "assistant", "content": "I have read the article. Please provide your question."}]
        inputs = self.template2ids(context)
        self.model(**inputs)
        mem_state = self.model.memory.export()

        outputs = []
        if query:
            for i,inst in enumerate(instruct):
                if i > 0:
                    self.model.memory.reset(**mem_state)
                sample = [
                        {"role": "user", "content": inst.format(question=query)}]
                inputs = self.template2ids(sample)
                res = self.ids2text(inputs, **generation_kwargs)
                outputs.append(res)
        else:
            sample = [
                    {"role": "user", "content": instruct[0]}]
            inputs = self.template2ids(sample)
            res = self.ids2text(inputs, **generation_kwargs)
            outputs.append(res)
        return outputs

    def ids2text(self, inputs, **generation_kwargs):
        outputs = self.model.generate(
            **inputs,
            **generation_kwargs,
            pad_token_id=self.tokenizer.eos_token_id)
        outputs = outputs[:, inputs["input_ids"].shape[1]:]
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        return outputs
    
    def determine_window_size(self, text):
        chunks = list(self.splitter.chunks(text))
        chunk_lengths = [self.tokenizer(chunk, return_length=True)['length'] for chunk in chunks]
        window_size = max(2048, int(np.mean(chunk_lengths)))
        print(window_size)
        return window_size

    def generate(self, prompts:Union[str, List[str]], batch_size:int=4, **generation_kwargs):
        all_outputs = []

        if isinstance(prompts, str):
            squeeze = True
            prompts = [prompts]
        else:
            squeeze = False

        # # window_size
        # combined_text = " ".join(prompts)
        # window_size = self.determine_window_size(combined_text)
        
        # # SelfExtend
        # SelfExtend.apply(self.model, 32, window_size, enable_flash_attention=True, flash_attention_impl="flash_attn")

        # model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        # minference_patch = MInference("inf_llm", model_name) # inf_llm # minference #streaming2
        # self.model=minference_patch(self.model)

        for i in range(0, len(prompts), batch_size):
            batch_prompts = []
            # import pdb; pdb.set_trace()
            for prompt in prompts[i: i + batch_size]:
                prompt = self.tokenizer.apply_chat_template([{"role":"user", "content": prompt}], tokenize=False, add_generation_prompt=True)
                batch_prompts.append(prompt)

            inputs = self.tokenizer(batch_prompts, add_special_tokens=False, return_tensors="pt", padding=True).to(self.model.device)
   
            outputs = self.model.generate(
                **inputs, 
                **generation_kwargs,
                pad_token_id=self.tokenizer.eos_token_id)
                    
            outputs = outputs[:, inputs["input_ids"].shape[1]:]

            # decode to string
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_outputs.extend(outputs)

        if squeeze:
            all_outputs = all_outputs[0]

        return all_outputs


def init_args(model_args, model_name, device):
    model_args_dict = model_args.to_dict()
    dtype = model_args_dict["dtype"]
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    if model_args_dict["device_map"] is None and not is_deepspeed_zero3_enabled():
        # device_map = {"": device}
        device_map = "auto"

    model_kwargs = {
        "cache_dir": model_args_dict["model_cache_dir"],
        "token": model_args_dict["access_token"],
        "device_map": model_args_dict["device_map"],
        "attn_implementation": model_args_dict["attn_impl"],
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": True,
        "load_in_4bit": model_args_dict["load_in_4bit"]
    }

    tokenizer_kwargs = {
        "cache_dir": model_args_dict["model_cache_dir"],
        "token": model_args_dict["access_token"],
        "padding_side": model_args_dict["padding_side"],
        "trust_remote_code": True,
    }
    return model_kwargs, tokenizer_kwargs

import random
import numpy as np
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    from args import ModelArgs
    from transformers import HfArgumentParser
    from accelerate import Accelerator

    seed_everything(42)
    parser = HfArgumentParser([ModelArgs])
    model_args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator(cpu=model_args.cpu)
    model_kwargs, tokenizer_kwargs = init_args(
        model_args, 
        model_args.gen_model, 
        accelerator.device)
    
    model = HuggingFaceModel(
        model_args.gen_model, 
        model_kwargs=model_kwargs,
        tokenizer_kwargs=tokenizer_kwargs)

    print(model.model.device)
    print(model.model)
