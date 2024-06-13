import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

from load_data import load_conll_data, load_i2b2_data
from prompt import llama3_prompt, mixtral_prompt
from utils import get_current_date_as_string


def compute(texts, model):
    if "llama-3" in model.lower():
        prompt_function = llama3_prompt
    elif "mixtral" in model.lower():
        prompt_function = mixtral_prompt

    inputs = [prompt_function(text) for text in texts]

    sampling_params = SamplingParams(
        temperature=0,  # Temperature set to 0 for deterministic outputs
        max_tokens=500,  # Maximum tokens to generate (necessary for Llama-3 models)
    )

    # Assumes 8 GPUs are available. If not, set tensor_parallel_size to the number of GPUs available.
    if "70b" in model.lower():
        llm = LLM(model=model, tensor_parallel_size=8, gpu_memory_utilization=0.80)
    else:
        llm = LLM(model=model, tensor_parallel_size=8)

    start = time.time()
    outputs = llm.generate(inputs, sampling_params)
    end = time.time()

    outputs_json = []
    for output in outputs:
        generated_text = output.outputs[0].text
        outputs_json.append(generated_text)

    return outputs_json, end - start


def run_model_on_dataset(model, output_dir, dataset, dataset_path):
    if dataset == "conll":
        texts = load_conll_data()
    elif dataset == "i2b2":
        texts = load_i2b2_data(dataset_path)

    results, times = compute(texts, model)

    results_file_path = os.path.join(
        output_dir,
        f"results_{get_current_date_as_string()}_{model}_zero-shot_{dataset}.json",
    )
    with open(results_file_path, "w") as file:
        json.dump(results, file)

    times_file_path = os.path.join(
        output_dir,
        f"times_{get_current_date_as_string()}_{model}_zero-shot_{dataset}.json",
    )
    with open(times_file_path, "w") as file:
        json.dump(times, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT model on CoNLL data")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use e.g. 'Meta-Llama-3-8b-Instruct'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to run the model on. Can be 'conll' or 'i2b2'",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        help="Path to the I2B2 dataset directory",
    )

    args = parser.parse_args()
    if args.prompt_type not in ["zero-shot", "one-shot"]:
        raise ValueError("prompt_type must be either 'zero-shot' or 'one-shot'")
    if args.dataset not in ["conll", "i2b2"]:
        raise ValueError("dataset must be either 'conll' or 'i2b2'")

    run_model_on_dataset(
        args.gpt_model, args.output_dir, args.dataset, args.dataset_path
    )
