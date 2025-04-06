import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class ModelArgs:
    strategy: Optional[str] = field(
        default='default_strategy',  # 默认值
        metadata={"help": "The strategy to use for some operation."}  # 帮助信息
    )
    model_cache_dir: str = field(
        default='',
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: str = field(
        default='',
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default="./data/", 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json file or glob to match a list of files.'},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    index_path: Optional[str] = field(
        default="",
        metadata={'help': 'Evaluation json file.'},
    )
    model_name_or_path: str = field(
        default='mistralai/Mistral-7B-Instruct-v0.2',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    access_token: Optional[str] = field(
        default='',
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={'help': 'The implementation of attention.'}
    )
    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    chat_template: str = field(
        default="llama-2",
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )

    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )
    cache_implementation: str = field(
        default=None,
        metadata={'help': 'use cache?'}
    )

    cache_backend: str = field(
        default=None,
        metadata={'help': 'cache backend'}
    )

    cache_nbits: int = field(
        default=None,
        metadata={'help': 'quant size'}
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={'help': 'quant size'}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )

    gen_model: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={'help': 'Model name or path for generation.'}
    )
    gen_max_new_tokens: Optional[int] = field(
        default=512,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    gen_do_sample: Optional[bool] = field(
        default=False,
        metadata={'help': 'Do sampling when decoding?'},
    )
    gen_temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    gen_top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    pipeline: str = field(
        default="recall-refine",
        metadata={'help': 'Pipeline name. {recall-refine, rag, direct}'}
    )
    note: str = field(
        default="",
        metadata={'help': 'experiment note'}
    )
    conv: bool = field(
        default=False,
        metadata={'help': 'Merge and unload LoRA?'},
    )



    def resolve_path(self, path):
        """Resolve any path starting with 'long-llm:' to relative path against data_root."""
        pattern = "long-llm:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def __post_init__(self):
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.output_dir = self.resolve_path(self.output_dir)

        if hasattr(self, "result_dir"):
            if self.result_dir is None: 
                self.result_dir = "tmp"