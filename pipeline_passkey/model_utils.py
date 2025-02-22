import logging
import os
import json
import pdb

logger = logging.getLogger("main")


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, chat_template):
    if "llama3" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif "longchat" in chat_template or "vicuna" in chat_template:
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "mistral" in chat_template.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif chat_template is None:
        return prompt

    else:
        logger.error(f"{chat_template} is unsupported.")
        raise NotImplementedError
    return prompt
