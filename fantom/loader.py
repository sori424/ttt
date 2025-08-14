import pandas as pd
import random

def set_beliefQA_multiple_choices(qa):
    if qa['question_type'].endswith(":inaccessible"):
        option_a = qa['wrong_answer']
        option_b = qa['correct_answer']
    else:
        option_a = qa['wrong_answer']
        option_b = qa['correct_answer']

    answer_goes_last = random.choice([True, False])
    if answer_goes_last:
        choices = [option_a, option_b]
        answer = 1
    else:
        choices = [option_b, option_a]
        answer = 0

    # option letters iterate over the alphabet
    option_letters = ["(" + chr(x) + ")" for x in range(ord('a'), len(choices) + ord('a'))]
    choices_text = ""
    for letter, option in zip(option_letters, choices):
        choices_text += "{} {}\n".format(letter, option)

    return choices_text, answer


def setup_fantom(df):
    """
    Flatten the dictionary and add short and full conversation context to each question.
    The result will be a list of questions and list of short or full inputs to be used as input for the models.
    """
    aggregation_target = "conversation"
    conversation_input_type ="full" #"The input type should have been the full conversation. It doesn't make sense to aggregate the scores over the full conversation when the input is not the full conversation"

    fantom_df_to_run = df

    total_num_q = 0
    for idx, _set in fantom_df_to_run.iterrows():
        total_num_q += len(_set['beliefQAs'])
        total_num_q += len(_set['answerabilityQAs_binary'])
        total_num_q += len(_set['infoAccessibilityQAs_binary'])
        if _set['factQA'] is not None:
            total_num_q += 1
        if _set['answerabilityQA_list'] is not None:
            total_num_q += 1
        if _set['infoAccessibilityQA_list'] is not None:
            total_num_q += 1

    inputs = []
    qas = []
    for idx, _set in fantom_df_to_run.iterrows():
        if conversation_input_type == "short":
            context = _set['short_context'].strip()
        elif conversation_input_type == "full":
            context = _set['full_context'].strip()
        
        set_id = _set['set_id']
        fact_q = _set['factQA']['question']
        fact_a = _set['factQA']['correct_answer']

        # Fact Question
        _set['factQA']['context'] = context
        input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, fact_q)
        _set['factQA']['input_text'] = input_text
        _set['factQA']['set_id'] = set_id
        qas.append(_set['factQA'])
        inputs.append(input_text)

        for _belief_qa in _set['beliefQAs']:
            # Belief Questions
            _belief_qa['context'] = context
            input_text = "{}\n\nQuestion: {}\nAnswer:".format(context, _belief_qa['question'])
            _belief_qa['input_text'] = input_text
            _belief_qa['set_id'] = set_id
            qas.append(_belief_qa)
            inputs.append(input_text)

            # Multiple Choice Belief Questions
            _mc_belief_qa = {**_belief_qa}
            choices_text, answer = set_beliefQA_multiple_choices(_mc_belief_qa)
            mc_question = "{}\n{}\n\nChoose an answer from above:".format(_belief_qa['question'], choices_text.strip())
            _mc_belief_qa['question'] = mc_question
            _mc_belief_qa['question_type'] = _mc_belief_qa['question_type'] + ":multiple-choice"
            _mc_belief_qa['choices_text'] = choices_text
            _mc_belief_qa['choices_list'] = choices_text.strip().split("\n")
            _mc_belief_qa['correct_answer'] = answer
            input_text = "{}\n\nQuestion: {}".format(context, mc_question)
            _mc_belief_qa['input_text'] = input_text
            qas.append(_mc_belief_qa)
            inputs.append(input_text)

        # Answerability List Questions
        _set['answerabilityQA_list']['fact_question'] = fact_q
        _set['answerabilityQA_list']['context'] = context
        input_text = "{}\n\nTarget: {}\nQuestion: {}\nAnswer:".format(context, fact_q, _set['answerabilityQA_list']['question'])
        _set['answerabilityQA_list']['input_text'] = input_text
        _set['answerabilityQA_list']['set_id'] = set_id
        if conversation_input_type == "full" and len(_set['answerabilityQA_list']['wrong_answer']) > 0:
            _set['answerabilityQA_list']['missed_info_accessibility'] = 'inaccessible'
        qas.append(_set['answerabilityQA_list'])
        inputs.append(input_text)

        # Answerability Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set['answerabilityQAs_binary'][0]['missed_info_accessibility']
            for _info_accessibility_qa in _set['answerabilityQAs_binary']:
                if _info_accessibility_qa['correct_answer'] != "yes":
                    missed_info_accessibility_for_full = 'inaccessible'

        for _answerability_qa in _set['answerabilityQAs_binary']:
            _answerability_qa['fact_question'] = fact_q
            _answerability_qa['context'] = context
            input_text = "{}\n\nTarget: {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, _answerability_qa['question'])
            _answerability_qa['input_text'] = input_text
            _answerability_qa['set_id'] = set_id
            if conversation_input_type == "full":
                _answerability_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
            qas.append(_answerability_qa)
            inputs.append(input_text)

        # Info Accessibility List Questions
        _set['infoAccessibilityQA_list']['fact_question'] = fact_q
        _set['infoAccessibilityQA_list']['fact_answer'] = fact_a
        _set['infoAccessibilityQA_list']['context'] = context
        input_text = "{}\n\nInformation: {} {}\nQuestion: {}\nAnswer:".format(context, fact_q, fact_a, _set['infoAccessibilityQA_list']['question'])
        _set['infoAccessibilityQA_list']['input_text'] = input_text
        _set['infoAccessibilityQA_list']['set_id'] = set_id
        if conversation_input_type == "full" and len(_set['infoAccessibilityQA_list']['wrong_answer']) > 0:
            _set['infoAccessibilityQA_list']['missed_info_accessibility'] = 'inaccessible'
        qas.append(_set['infoAccessibilityQA_list'])
        inputs.append(input_text)

        # Info Accessibility Binary Questions
        if conversation_input_type == "full":
            missed_info_accessibility_for_full = _set['infoAccessibilityQAs_binary'][0]['missed_info_accessibility']
            for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
                if _info_accessibility_qa['correct_answer'] != "yes":
                    missed_info_accessibility_for_full = 'inaccessible'

        for _info_accessibility_qa in _set['infoAccessibilityQAs_binary']:
            _info_accessibility_qa['fact_question'] = fact_q
            _info_accessibility_qa['fact_answer'] = fact_a
            _info_accessibility_qa['context'] = context
            input_text = "{}\n\nInformation: {} {}\nQuestion: {} Answer yes or no.\nAnswer:".format(context, fact_q, fact_a, _info_accessibility_qa['question'])
            _info_accessibility_qa['input_text'] = input_text
            _info_accessibility_qa['set_id'] = set_id
            if conversation_input_type == "full":
                _info_accessibility_qa['missed_info_accessibility'] = missed_info_accessibility_for_full
            qas.append(_info_accessibility_qa)
            inputs.append(input_text)

    inputs = inputs
    flattened_fantom = qas

    return inputs, flattened_fantom