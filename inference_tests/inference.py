import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer
from llama_recipes.inference.model_utils import load_model, load_peft_model

import pandas as pd
import random 

def main(
    model_name,
    prompts,
    size: int=1000,
    print_output: bool=False,
    print_times: bool=False,
    seed=0,
    peft_model=None,
    quantization=False,
    use_fast_kernels=False,
    max_padding_length: int=None, 
    max_new_tokens =100,
    do_sample: bool=True,
    top_p: float=1.0,
    temperature: float=1.0, 
    min_length: int=None,
    use_cache: bool=True,
    top_k: int=50,
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, 
   **kwargs
):
    if prompts is not None:
        assert os.path.exists(prompts), f"Provided Prompts do not exist: {prompts}"
  
    # Seed for Reproducability
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Get prompts
    instructions = pd.read_csv(prompts)
    instructions = instructions.values.tolist()
    randomInstructions = random.sample(instructions, 1000)
    
    # Load Model
    model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    # Implement Later
    if use_fast_kernels:
       pass

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    e2e_inference_times = []

    for instruction in randomInstructions:
        batch = tokenizer(instruction, padding='max_length', truncation=True, max_length=max_padding_length, return_tensors="pt")
        batch = {k: v.to("cuda") for k, v in batch.items()}
        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                **kwargs 
            )
        e2e_inference_time = (time.perf_counter()-start)*1000
        e2e_inference_times.append(e2e_inference_time)

        if print_times:
            print(f"Inference time: {e2e_inference_time}")
        if print_output:
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Model output:\n{output_text}")
    average_time = sum(e2e_inference_times) / len(e2e_inference_times)
    print(f"The average end-to-end inference time over {size} prompts is {average_time}")

if __name__ == "main":
    fire.Fire(main)