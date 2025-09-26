from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto').to('cuda')

    return tokenizer, model

def gen_text(model, tokenizer, input_text):
    input = tokenizer(input_text, return_tensors='pt').to('cuda')
    output_tokens = model.generate(
        **input,
        max_new_tokens = 40,
        num_return_sequences=1,
        do_sample = True,
        top_p = 0.95,
        temperature = 0.8
    )

    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    return generated_text


def gen_chat_template_system(model, tokenizer, input_text, rule):

    message = [
        {"role": "system", "content": f"You are a helpful assistant. Use following rule when answering questions: {rule}"},
        {"role": "user", "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input = tokenizer(prompt, return_tensors='pt').to('cuda')

    output_tokens = model.generate(
        **input,
        max_new_tokens = 100,
        num_return_sequences=1,
        do_sample = True,
        top_p = 0.95,
        temperature = 0.8
    )

    generated = output_tokens[0][input["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)


    return generated_text

def gen_chat_template(model, tokenizer, input_text):

    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    input = tokenizer(prompt, return_tensors='pt').to('cuda')

    output_tokens = model.generate(
        **input,
        max_new_tokens = 100,
        num_return_sequences=1,
        do_sample = True,
        top_p = 0.95,
        temperature = 0.8
    )

    generated = output_tokens[0][input["input_ids"].shape[-1]:]
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)


    return generated_text
