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

def load_existing_responses(args,input_len):
    short_model_name = args.model_name.split("/")[-1]  
    prompt_name = args.rules.split("/")[-1].replace('.json','')    
    responses_filename = f"model_responses_{short_model_name}_{input_len}-instances_{args.task_name}_cot-{args.use_cot}_sp-{args.system_prompt}_{prompt_name}_{args.effort}.jsonl" 
    
    response_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)
    search_dict = {}
    with open(response_filename_path, 'r') as f:
        for line in f:
            resp = json.loads(line)
            story_idx = resp["story_index"]
            w_story_idx = resp["within_story_index"]
            rule_idx = resp["rule_index"]
            try:
                search_dict[story_idx]
            except KeyError:
                search_dict[story_idx] = {}
            try:
                search_dict[story_idx][w_story_idx].append(rule_idx)
            except KeyError:
                search_dict[story_idx][w_story_idx] = [rule_idx]

    return search_dict


def run_inference_rules_general(args, inputs, model, tokenizer):
    target_data = inputs
    model_responses = {}

    # load previous responses
    search_dict = load_existing_responses(args, len(inputs))
    short_model_name = args.model_name.split("/")[-1]  
    rules = json.load(open(args.rules, 'r'))
    prompt_name = args.rules.split("/")[-1].replace('.json','')    
    responses_filename = f"model_responses_{short_model_name}_{len(inputs)}-instances_{args.task_name}_cot-{args.use_cot}_sp-{args.system_prompt}_{prompt_name}_{args.effort}.jsonl" 
    response_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)
    logging.info(f"Generating responses...")
    for idx, item in enumerate(tqdm(target_data)):
        for rule in rules:
            meta = item.get('meta_data', {})
            story_idx = meta.get('story_index')
            w_story_idx = meta.get('within_story_index')
            try:
                if rule['index'] in search_dict[story_idx][w_story_idx]:
                    logging.info(f"Rule {rule['index']} for story {story_idx} and within story {w_story_idx} already exists. Skipping.")
                    continue
            except KeyError:
                pass
            input_prompt = 'Story: ' + item['story'] + '\n\nQuestion: ' + item['question']   
            if args.system_prompt:
                response = gen_chat_template_system(model, tokenizer, input_prompt, rule['natural_language'], args.effort)
            else:
                if args.use_nl:
                    # Use natural language rule
                    input_prompt += '\n\n' + rule['natural_language']
                else:
                    # TODO: implement me!!!!
                    continue
                # This is only for checkpointing
                #if idx <= last_idx:
                #    continue
                response = gen_chat_template(model, tokenizer, input_prompt, args.effort)
                
            response = parse_response(response)

            try:
                model_responses[story_idx]
            except KeyError:
                model_responses[story_idx] = {}
            try:
                model_responses[story_idx][w_story_idx] 
            except KeyError:
                model_responses[story_idx][w_story_idx] = {}

            model_responses[story_idx][w_story_idx][rule['index']] = response
            #"response":response, "rule_index":rule['index'], "story_index":meta.get('story_index'), "within_story_index":meta.get('within_story_index') })

            # save the model responses in a file on the fly
            with open(response_filename_path, 'a') as f:
                json.dump({'story_index': meta.get('story_index'), 'within_story_index':meta.get('within_story_index'), 'rule_index':rule['index'], 'input_prompt': input_prompt, 'response': response}, f)
                f.write("\n")

    #assert len(model_responses) == len(target_data)

    return model_responses


def run_inference_rules_file(args, inputs, model, tokenizer):
    target_data = inputs
    model_responses = []

    # load previous responses
    search_dict = load_existing_responses(args)
    short_model_name = args.model_name.split("/")[-1]  
    prompt_name = args.rules.split("/")[-1].replace('.json','')    
    responses_filename = f"model_responses_{short_model_name}_{len(inputs)}-instances_{args.task_name}_cot-{args.use_cot}_sp-{args.system_prompt}_{prompt_name}_{args.effort}.jsonl" #
    response_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)
    logging.info(f"Generating responses...")
    for idx, item in enumerate(tqdm(target_data)):
        input_prompt = 'Story: ' + item['story'] + '\n\nQuestion: ' + item['question']
        # This is only for checkpointing
        #if idx <= last_idx:
        #    continue
        if args.system_prompt:
            response = gen_chat_template_system(model, tokenizer, input_prompt, item['natural_language'], args.effort)
        else:
            if args.use_nl:
                # Use natural language rule
                input_prompt += '\n\nRule to answer this question: ' + item['natural_language']
            else:
                try:
                    input_prompt += '\n\nRule to answer this question: ' + item['typed_variables'] + '\n' + item['rule_if'] + '\n' + item['rule_then']
                except TypeError:
                    logging.error(f"Error in instance {item['index']}! Expected to find typed_variables, rule_if, and rule_then but got:")
                    logging.error(f"{item['typed_variables']}, {item['rule_if']}, and {item['rule_then']} ")
                    logging.error(f"Skipping instance {item['index']}!")
                    continue
            response = gen_chat_template(model, tokenizer, input_prompt, args.effort)
            
            
        response = parse_response(response)
        model_responses.append(response)

        # save the model responses in a file on the fly
        with open(response_filename_path, 'a') as f:
            json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
            f.write("\n")

    #assert len(model_responses) == len(target_data)

    return model_responses

def run_inference(args, inputs, model, tokenizer):  
    target_data = inputs
    model_responses = []

    # check if the file exists
    #last_idx, model_responses, response_filename_path = get_last_savepoint(args)
    prompt_name = args.prompt.replace(".txt","")
    short_model_name = args.model_name.split("/")[-1]  
    responses_filename = f"model_responses_{short_model_name}_{len(inputs)}-instances_{args.task_name}_cot-{args.use_cot}_sp-{args.system_prompt}_{prompt_name}_{args.effort}.jsonl" #
    response_filename_path = os.path.join(EVAL_DIR_PATH, responses_filename)    
    logging.info(f"Generating responses...")
    for idx, item in enumerate(tqdm(target_data)):

        input_prompt = 'Story: ' + item['input_text']
        #if idx <= last_idx:
        #    continue

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
        
        response = gen_chat_template(model, tokenizer, input_prompt, args.effort)
        response = parse_response(response)
        model_responses.append(response)

        # save the model responses in a file on the fly
        with open(response_filename_path, 'a') as f:
            json.dump({'index': idx, 'input_prompt': input_prompt, 'response': response}, f)
            f.write("\n")

    #assert len(model_responses) == len(target_data)

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
    
    parser.add_argument('--general-rules', # use rules or not
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Individual or general rules (will apply every rule to every case!',
    )
    
    parser.add_argument('--use-nl', # use nl rule or if thenn else
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='whether to use rules or not',
    )
    
    parser.add_argument('--debugging', # or no-use-tt
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Subsample instances etc. to make things more efficient or not',
    )
    
    parser.add_argument('--system-prompt', # add the rule to the system prompt or not
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='add the rule to the system prompt or not',
    )
    
    parser.add_argument('--effort', # reasoning effort for gpt 
                        type=str,
                        default="low",
                        help='reasoning effort for GPT',
    )
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_name)

    # Use separate inputs
    inputs = json.load(open(os.path.join(DATA_DIR_PATH, f'{args.task_name}/{args.task_name}_flattened.json'), 'r'))
    if args.debugging:
        inputs = random.sample(inputs, 10)  ## for test 
 
    if args.use_rules:

        if args.general_rules:
            model_responses = run_inference_rules_general(args, inputs, model, tokenizer)
        else:   
            if args.debugging:
                rule_inputs = json.load(open(args.rules, 'r'))
                rule_inputs = random.sample(rule_inputs, 10)  ## for test 
            # TODO: change this to map the rule to the original flattened file.
            model_responses = run_inference_rules_file(args, rule_inputs, model, tokenizer)
    else:
        model_responses = run_inference(args, inputs, model, tokenizer)


    short_model_name = args.model_name.split("/")[-1]  
    prompt_name = args.prompt.replace(".txt","")
    if args.use_rules:
        prompt_name = args.rules.split("/")[-1].replace('.json','')
    fname = f"{len(inputs)}-instances_{short_model_name}_{args.task_name}_cot-{args.use_cot}_sp-{args.system_prompt}_nl-{args.use_nl}_{prompt_name}_{args.effort}"
    summary_file = f"./results/summary_{fname}.txt"
    
    if args.general_rules:
        report = evaluate_bigtom_general_rules(inputs, model_responses, summary_file)
    else:
        report = evaluate_bigtom(inputs, model_responses, summary_file)
    
    with open(f'./results/report_{fname}', 'w') as f:
        json.dump(report, f, indent=4)


if __name__ == '__main__':
    main()
