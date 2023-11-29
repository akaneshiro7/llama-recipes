import fire
import os
import random
import time

import torch
from vllm import LLM
from vllm import LLM, SamplingParams

from carbontracker.tracker import CarbonTracker, CarbonTrackerManual

import pandas as pd
import json

def load_model(model_name, tp_size=1):

    llm = LLM(model_name, tensor_parallel_size=tp_size)
    return llm

def main(
    model,
    max_new_tokens=100,
    prompts=None,
    top_p=0.9,
    temperature=0.8,
    size=250,
    seed=0,
    print_outputs=False,
    print_times=False,
    output_file=None
):
    if not output_file:
        print("No output file specified")
        return
    if prompts is not None:
        assert os.path.exists(prompts), f"Provided Prompts do not exist: {prompts}"
    
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Get prompts
    with open(prompts, 'r') as f:
        instructions = [line.strip() for line in f]
    
    info = {}

    sampling_param = SamplingParams(top_p=top_p, temperature=temperature, max_tokens=max_new_tokens)
    tracker = CarbonTrackerManual(epochs=2, monitor_epochs=1, update_interval=0.1,
        components='all', epochs_before_pred=1, verbose=0)
    tracker.tracker.pue_manual=1
    tracker.intensity_updater.ci_manual = 100
    time.sleep(5)
    
    for i, instruction in enumerate(instructions):
        tracker.epoch_start()
        print(f"Prompt {i}")

        outputs = model.generate(instruction, sampling_params=sampling_param)

        [energy, co2] = tracker.epoch_end()

        info[str(instruction)] = {
            "Energy": energy,
            "CO2": co2
        }


    with open(output_file, 'w') as f:
        json.dump(info, f, indent=4)

def run_script(
    model_name: str,
    peft_model=None,
    tp_size=1,
    max_new_tokens=100,
    prompts=None,
    top_p=1.0,
    temperature=1.0,
    size=250,
    seed=0,
    print_outputs=False,
    print_times=False,
    output_file=None,
   **kwargs
):

    model = load_model(model_name, tp_size)
    main(model, max_new_tokens=max_new_tokens, prompts=prompts, top_p=top_p, temperature=temperature, size=size, seed=seed, print_outputs=print_outputs, print_times=print_times, output_file=output_file)

fire.Fire(run_script)
