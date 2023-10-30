import fire
import os
import random
import time

import torch
from vllm import LLM
from vllm import LLM, SamplingParams

import pandas as pd

def load_model(model_name, tp_size=1):

    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm

def main(
    model,
    max_new_tokens=100,
    prompts=None,
    top_p=0.9,
    temperature=0.8,
    size=1000,
    seed=0,
    print_outputs=False,
    print_times=False
):
    if prompts is not None:
        assert os.path.exists(prompts), f"Provided Prompts do not exist: {prompts}"
    
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    instructions = pd.read_csv(prompts)
    instructions = instructions.values.tolist()
    randomInstructions = random.sample(instructions, 1000)

    e2e_inference_times = []

    for instruction in randomInstructions:
        sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)
        start = time.perf_counter
        outputs = model.generate(instruction, sampling_params=sampling_param)
        e2e_inference_time = (time.perf_counter()-start)*1000
        e2e_inference_times.append(e2e_inference_time)
    
        if print_times:
            print(f"Inference time: {e2e_inference_time}")
        if print_outputs:
            print(f"model output:\n {instruction} {outputs[0].outputs[0].text}")

    average_time = sum(e2e_inference_times) / len(e2e_inference_times)
    print(f"The average end-to-end inference time over {size} prompts is {average_time}")

def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1,
    max_new_tokens=100,
    prompts=None,
    top_p=1.0,
    temperature=1.0,
    size=1000,
    seed=0,
    print_outputs=False,
    print_times=False
   **kwargs
):

    model = load_model(model_name, tp_size)
    main(model, max_new_tokens=max_new_tokens, prompts=prompts, top_p=top_p, temperature=temperature, size=size, seed=seed, print_outputs=print_outputs, print_times=print_times)

if __name__ == "__main__":
    fire.Fire(run_script)