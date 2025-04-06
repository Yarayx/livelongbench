import logging
logger = logging.getLogger("main")

import eval.passkey_utils.passkey_main as passkey_main
import eval.passkey_utils.passkey_utils as passkey_utils
import inference as inference


def eval_passkey_retrieval(config):
    raw_exp_results = passkey_main.prepare_passkey_retrieval_input(config)
    eval_params = config['eval_params']
    pipeline_params = config['pipeline_params']

    logger.info(f"Starting evaluation via {pipeline_params['method']}")

    model,tokenizer= inference.initialize_model_tokenizer(pipeline_params=config['pipeline_params'])
    longest_input = raw_exp_results[-1]['full_input']

    passkey_utils.check_if_out_of_context_window(longest_input=longest_input, model_max_len=config['pipeline_params']['model_max_len'], tokenizer=tokenizer,out_of_max_len_allowed=config['pipeline_params']['out_of_max_len_allowed'])

    batch_size = config['pipeline_params']['batch_size']
    batched_raw_exp_results = [raw_exp_results[i:i + batch_size] for i in
                                range(0, len(raw_exp_results), batch_size)]

    for i, one_batch in enumerate(batched_raw_exp_results):
        batched_input = [i['full_input'] for i in one_batch]
        batched_responses = inference.batch_generate(batched_input=batched_input, model=model, tokenizer=tokenizer,max_new_tokens=config['eval_params']['max_new_tokens'])

        for one_exp_results, one_response in zip(one_batch, batched_responses):
            one_exp_results['response'] = one_response

        logger.info(f'Finished evaluating batch {i + 1}/{len(batched_raw_exp_results)} (batch_size = {batch_size}).')
    logger.info(f'Finished evaluating all {len(batched_raw_exp_results)} batches (batch_size = {batch_size}).')

    processed_results, raw_results = passkey_utils.process_raw_exp_results(raw_exp_results=raw_exp_results, metrics=eval_params['eval_metrics'])
    logger.info('raw_exp_results processed.')

    return processed_results, raw_results
