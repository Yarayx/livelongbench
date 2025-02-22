import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models import init_args, HuggingFaceModel
from src.utils import *
from typing import Dict, Union, List, Optional
from itertools import chain
# from src import apply_chat_template
from src.models_m import HuggingFaceModel as HuggingFaceModel_m
from src.models_s import HuggingFaceModel as HuggingFaceModel_s
from src.models_ss import HuggingFaceModel as HuggingFaceModel_ss
from src.models_ms import HuggingFaceModel as HuggingFaceModel_ms
from src.models_mss import HuggingFaceModel as HuggingFaceModel_mss
prompts = {
  "key": "You are provided with a long article. Your task is to generate a concise summary by listing the key points of the long article.\n\n### Instructions:\n\n1. Long Article: {context}\n2. Output: Generate a list of key points, each separated by a newline, with numeric order.\n\n### Requirements:\n\n- The key points should be short and high-level.\n- Ensure that the key points convey the most important information and main events of the long article.\n",
  "span": "You are given a long article and a question. After a quick read-through, you have a rough memory of the article. To answer the question effectively, you need to recall and extract specific details from the article. Your task is to find and retrieve the relevant clue texts from the article that will help answer the question.\n\n### Inputs:\n- **Long Article:** {context}\n- **Question:** {input}\n\n### Requirements:\n1. You have a general understanding of the article. Your task is to identify and extract one or more specific clue texts from the article that are relevant to the question.\n2. Output only the extracted clue texts. For multiple sentences, separate them with a newline.\n",
  "sur": "You are provided with a long article and a question. After a quick read-through, you have a rough memory of the article. To better answer the question, you need to recall specific details within the article. Your task is to generate precise clue questions that can help locate the necessary information.\n\n### Inputs:\n- **Long Article:** {context}\n- **Question:** {input}\n\n### Requirements:\n1. You have a general understanding of the article. Your task is to write one or more precise clue questions to search for supporting evidence in the article.\n2. Output only the clue questions. For multiple questions, separate them with a newline.\n ",
  "qa": "You are given a {ctx_type}. You're required to read the {ctx_type} and answer the questions.\n\nNow the {ctx_type} begins. \n\n{context}\n\nNow the {ctx_type} ends.\n\nAnswer the following questions.\n\n{input}",
  "hyde": "Please write a passage to answer the question.\nQuestion: {input}\nPassage:",
  "clues": "Please write some clues to answer the question.\nQuestion: {input}\nclues:"
}

_prompt = """You are provided with a long article. Read the article carefully. After reading, you will be asked to perform specific tasks based on the content of the article.

Now, the article begins:
- **Article Content:** {context}

The article ends here.

Next, follow the instructions provided to complete the tasks."""

# _instruct_sur = """
# You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to generate specific clues that will help locate the necessary information within the article.

# ### Question: {question}
# ### Instructions:
# 1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
# 2. The clues can be in the form of precise surrogate questions that clarify the original question or text spans that will assist in answering the question.
# 3. Only output the clues. If there are multiple clues, separate them with a newline."""

_instruct_sur = """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to generate precise clue questions that can help locate the necessary information.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of precise surrogate questions that clarify the original question.
3. Only output the clues. If there are multiple clues, separate them with a newline."""

_instruct_span = """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to identify and extract one or more specific clue texts from the article that are relevant to the question.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of text spans that will assist in answering the question.
3. Only output the clues. If there are multiple clues, separate them with a newline."""

_instruct_qa = """
You are given a question related to the article. Your task is to answer the question directly.

### Question: {question}
### Instructions:
Provide a direct answer to the question based on the article's content. Do not include any additional text beyond the answer."""

_instruct_sum = """
Your task is to create a concise summary of the long article by listing its key points. Each key point should be listed on a new line and numbered sequentially.

### Requirements:

- The key points should be brief and focus on the main ideas or events.
- Ensure that each key point captures the most critical and relevant information from the article.
- Maintain clarity and coherence, making sure the summary effectively conveys the essence of the article.
"""

def get_pre_cached_index(path, device):
    rtn = {}
    for file in os.listdir(path):
        if file.endswith(".json"):
            _id = file.split(".")[0]
            if os.path.exists(os.path.join(path, f"{_id}.bin")):
                _index = FaissIndex(device)
                _index.load(os.path.join(path,  f"{_id}.bin"))
            else:
                _index = None
            corpus = load_json(os.path.join(path, f"{_id}.json"))
            rtn[_id] = {"index": _index, "corpus": corpus}
    return rtn

def get_pipeline(model_args, device="cpu", **kwargs):
    model_kwargs, tokenizer_kwargs = init_args(
        model_args, 
        model_args.gen_model, device)

    model_args_dict = model_args.to_dict()

    pipeline_name = model_args_dict["pipeline"]
    strategy_name = model_args_dict["strategy"]
    index_path = model_args_dict["index_path"]

    ### initialize generation model 
    
    if pipeline_name in ["longllm", "LongLive"]:
        gen_model_name = model_args_dict["gen_model"]
        
        print(model_kwargs)
 
        if strategy_name == "minf":
            gen_model = HuggingFaceModel_m(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
        elif strategy_name == "selfE":
            gen_model = HuggingFaceModel_s(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
        elif strategy_name == "selfEs":
            gen_model = HuggingFaceModel_ss(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
        elif strategy_name == "minf_selfE":
            gen_model = HuggingFaceModel_ms(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
        elif strategy_name == "minf_selfEs":
            gen_model = HuggingFaceModel_mss(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )
        else:
            gen_model = HuggingFaceModel(
                    gen_model_name,
                    model_kwargs=model_kwargs,
                    tokenizer_kwargs=tokenizer_kwargs
                )

        generation_kwargs = {}
        if model_args_dict["gen_max_new_tokens"]:
            generation_kwargs["max_new_tokens"] = model_args_dict["gen_max_new_tokens"]
        if model_args_dict["gen_do_sample"]:
            generation_kwargs["do_sample"] = model_args_dict["gen_do_sample"]
        if model_args_dict["gen_temperature"]:
            generation_kwargs["temperature"] = model_args_dict["gen_temperature"]
        if model_args_dict["gen_top_p"]:
            generation_kwargs["top_p"] = model_args_dict["gen_top_p"]
        if model_args_dict["cache_implementation"]:
            generation_kwargs["cache_implementation"] = model_args_dict["cache_implementation"]
            generation_kwargs["cache_config"] ={
                "backend": model_args_dict["cache_backend"], 
                "nbits": model_args_dict["cache_nbits"]}

    ### initialize pipeline
    if pipeline_name == "longllm":
        pipeline = LongLLMPipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs
        )
    elif pipeline_name == "LongLive":
        pipeline = LongLivePipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs
        )
    else:
        raise NotImplementedError

    return pipeline  

  
class LongLLMPipeline:
    def __init__(self, generator: Union[HuggingFaceModel], generation_kwargs:Dict={}):
        self.generator = generator
        self.generation_kwargs = generation_kwargs
        self.reset()

    def reset(self):
        # internal attributes that facilitates inspection
        if self.generator.model_name_or_path.find("ultragist") != -1:
            self.generator.model.memory.reset()

    def __call__(self, context:str, question:str, prompt:str, cache_id="", conv=False):
        """
        Directly answer the question based on the context using the generator.
        """
        self.reset()
        if question:
            answer_prompt = prompt.format(input=question, context=context)
        else:
            answer_prompt = prompt.format(context=context)

        answer_output = self.generator.generate(answer_prompt, **self.generation_kwargs)            
        answer_output = answer_output.replace("</s>", "")
        return answer_output, ""  

class LongLivePipeline:
    def __init__(self, generator: Union[HuggingFaceModel], generation_kwargs:Dict={}):
        self.generator = generator
        self.generation_kwargs = generation_kwargs
        self.reset()

    def reset(self):
        # internal attributes that facilitates inspection
        if self.generator.model_name_or_path.find("ultragist") != -1:
            self.generator.model.memory.reset()

    def __call__(self, context:str, cache_id="", conv=False):
        """
        Directly answer the question based on the context using the generator.
        """
        self.reset()
        answer_output = self.generator.generate(context, **self.generation_kwargs)
        # answer_output = answer_output.replace("</s>", "")
        return answer_output, ""   

        

if __name__ == "__main__":
    from args import ModelArgs
    from transformers import HfArgumentParser
    from accelerate import Accelerator
    

    parser = HfArgumentParser([ModelArgs])
    model_args = parser.parse_args_into_dataclasses()[0]
    accelerator = Accelerator(cpu=model_args.cpu)
    device = accelerator.device
    index_cache = get_pre_cached_index(model_args.index_path, device)
    print(f"{len(index_cache)} indices loaded.")
    pipe = get_pipeline(model_args, device)
    pipe.retriever._index = index_cache["debeb514d8e4b8d2e47b5a67bc22126d"]["index"]
    retrieval_corpus = index_cache["debeb514d8e4b8d2e47b5a67bc22126d"]["corpus"]
    topk_scores, topk_indices = pipe.retriever.search(
        queries="[This book] delivers clear-headed coverage of the life and loves of our favorite literary riddle.", hits=3)
    topk_indices = sorted([x for x in topk_indices[0].tolist() if x > -1])

    retrieval_results = [retrieval_corpus[i].strip() for i in topk_indices]
    knowledge = "\n\n".join(retrieval_results)
    print(knowledge)    





