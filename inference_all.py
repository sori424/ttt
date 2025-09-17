import os
import json
import argparse
import random
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import evaluate
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from llm_utils import *
from eval_utils import *
import pandas as pd


DATA_DIR_PATH = './data'
EVAL_DIR_PATH = './results'
RANDOM_SEED = 99
random.seed(RANDOM_SEED)

def parse_response(response):
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    elif "Choose an answer from above:" in response:
        response = response.split("Choose an answer from above:")[-1].strip()

    return response

def get_last_savepoint(args):
    short_model_name = args.model_name.split("/")[-1]  
    prompt_name = args.prompt.replace(".txt","")
    responses_filename = f"model_responses_{short_model_name}_{args.task_name}_cot-{args.use_cot}_{prompt_name}.jsonl" # jsonl
    model_responses_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)

    # check if model outputs file exists
    if os.path.exists(model_responses_filename_path):
        logging.info(f"File {model_responses_filename_path} exists. Reading responses from file...")
        df = pd.read_json(model_responses_filename_path, lines=True)
        if len(df) > 0:
            last_idx = df.iloc[-1]['index']
            model_responses = df['response'].tolist()
        else:
            last_idx = -1
            model_responses = []
    else:
        last_idx = -1
        model_responses = []
    
    return last_idx, model_responses, model_responses_filename_path

def run_inference_rules(args, inputs, model, tokenizer):
    
    target_data = inputs
    model_responses = []

    # check if the file exists
    #last_idx, model_responses, response_filename_path = get_last_savepoint(args)
    logging.info(f"Generating responses...")
    for idx, item in enumerate(tqdm(target_data)):
        input_prompt = 'Story: ' + item['story'] + '\n\nQuestion: ' + item['question']
        if args.use_nl:
            # Use natural language rule
            input_prompt += '\n\nRule to answer this question: ' + item['natural_language']
        else:
            input_prompt += '\n\nRule to answer this question: ' + item['typed_variables'] + '\n' + item['rule_if'] + '\n' + item['rule_then']
        
        if idx <= last_idx:
            continue
        
        response = gen_chat_template(model, tokenizer, input_prompt)
        response = parse_response(response)
        model_responses.append(response)

        # save the model responses in a file on the fly
        with open(response_filename_path, 'a') as f:
            json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
            f.write("\n")

    assert len(model_responses) == len(target_data)

    return model_responses

def run_inference(args, inputs, model, tokenizer):
    
    target_data = inputs
    model_responses = []

    # check if the file exists
    #last_idx, model_responses, response_filename_path = get_last_savepoint(args)
    logging.info(f"Generating responses...")
    for idx, item in enumerate(tqdm(target_data)):

        input_prompt = 'Story: ' + item['input_text']
        if idx <= last_idx:
            continue

        if args.use_cot:
            cot_input_prompt = input_prompt + " Let's think step by step."

            cot_response = gen_text(model, tokenizer, cot_input_prompt)
            cot_response = parse_response(cot_response)
            input_prompt = cot_input_prompt + " " + cot_response + "\n\nTherefore, the answer is:"

        if args.use_tt:
            with open(f"./prompt/{args.prompt}", "r", encoding="utf-8") as file:
                tt_text = file.read()
            tt_input_prompt = f"{tt_text}\n\n{input_prompt}"
            input_prompt = tt_input_prompt
        
        response = gen_chat_template(model, tokenizer, input_prompt)
        response = parse_response(response)
        model_responses.append(response)

        # save the model responses in a file on the fly
        with open(response_filename_path, 'a') as f:
            json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
            f.write("\n")

    assert len(model_responses) == len(target_data)

    return model_responses


def main():
    parser = argparse.ArgumentParser(description='arguments for evaluation of FANToM dataset with models')
    parser.add_argument('--model_name',
                        type=str,
                        help='name of the model to run evaluation',
    )
    parser.add_argument('--task_name',
                    type=str,
                    help='name of the task to run evaluation',
    )

    parser.add_argument('--use-cot', # or no-use-cot
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='whether to use cot or not',
    )

    parser.add_argument('--prompt', # or no-use-tt
                        type=str,
                        default="tt_bigtom.txt",
                        help='path to the prompt',
    )
    
    parser.add_argument('--rules', # or no-use-tt
                        type=str,
                        default="results-backward_belief-false_belief-0.json",
                        help='Files with all the rules.',
    )
    
    parser.add_argument('--use-tt', # or no-use-tt
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='whether to use tt or not',
    )
    
    parser.add_argument('--use-rules', # use rules or not
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='whether to use rules or not',
    )
    
    parser.add_argument('--nl-rule', # use nl rule or if thenn else
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='whether to use rules or not',
    )
    
    parser.add_argument('--debugging', # or no-use-tt
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Subsample instances etc. to make things more efficient or not',
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)
    

    
    if args.debugging:
        inputs = random.sample(inputs, 200)  ## for test
        
    if args.use_rules:
        inputs = json.load(open(args.rules, 'r'))
        model_responses = run_inference_rules(args, inputs, model, tokenizer)
    else: 
        inputs = json.load(open(os.path.join(DATA_DIR_PATH, f'{args.task_name}/{args.task_name}_flattened.json'), 'r'))
        model_responses = run_inference(args, inputs, model, tokenizer)

    if args.task_name == 'fantom':
        conversation_input_type = 'full'
        aggregation_target = 'set'

        qas = evaluate_fantom(inputs, model_responses)
        report = run_reports(qas, aggregation_target, conversation_input_type, args.model_name)
        
    elif args.task_name == 'bigtom':
        short_model_name = args.model_name.split("/")[-1]  
        summary_file = f"./results/summary_{len(inputs)}-instances_{short_model_name}_{args.task_name}_cot-{args.use_cot}_{args.prompt}"
        report = evaluate_bigtom(inputs, model_responses, summary_file)

############ TODO: here you add more task, add evaluate function on utils.py
        
    elif args.task_name == 'hitom':
        report = evaluate_hitom(inputs, model_responses)

    short_model_name = args.model_name.split("/")[-1]  
    prompt_name = args.prompt.replace(".txt","")
    if use_rules:
        prompt_name = args.rules.split("/")[-1].replace('.json','')
    with open(f'./results/report_{len(inputs)}-instances_{short_model_name}_{args.task_name}_cot-{args.use_cot}_{prompt_name}.json', 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == '__main__':
    main()
