import argparse
import json
import os
import time

import openai

from load_data import load_conll_data
from prompt import zero_shot_prompt, one_shot_prompt, system_prompt
from utils import get_current_date_as_string


API_KEY = os.getenv("OPENAI_API_KEY")
if API_KEY is None:
    raise ValueError("Environment variable 'OPENAI_API_KEY' is not set.")

client = openai.OpenAI(api_key=API_KEY)


def compute(texts, gpt_model, prompt_type):
    print(f"Running GPT model: {gpt_model} with prompt type: {prompt_type}")
    results = []
    times = []
    for idx, text in enumerate(texts):
        if idx % 100 == 0:
            print(f"Processing text number: {idx}")
        if prompt_type == "zero-shot":
            prompt_function = zero_shot_prompt
        elif prompt_type == "one-shot":
            prompt_function = one_shot_prompt
        else:
            raise Exception()

        start = time.time()
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": f"{system_prompt}",
                    },
                    {
                        "role": "user",
                        "content": prompt_function(text),
                    },
                ],
                model=gpt_model,
                seed=12,  # Randomly chosen seed for reproducibility
                temperature=0,  # Temperature set to 0 for deterministic outputs
            )
            results.append(chat_completion.choices[0].message.content)
        except:
            results.append(None)

        end = time.time()
        times.append(end - start)

    return results, times


def run_gpt_on_conll(gpt_model, prompt_type, output_dir):
    texts = load_conll_data()

    results, times = compute(texts, gpt_model, prompt_type)

    results_file_path = os.path.join(
        output_dir,
        f"results_{get_current_date_as_string()}_{gpt_model}_{prompt_type}_conll.json",
    )
    with open(results_file_path, "w") as file:
        json.dump(results, file)

    times_file_path = os.path.join(
        output_dir,
        f"times_{get_current_date_as_string()}_{gpt_model}_{prompt_type}_conll.json",
    )
    with open(times_file_path, "w") as file:
        json.dump(times, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GPT model on CoNLL data")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="GPT model name to use e.g. 'gpt-3.5-turbo'",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        help="Type of prompt to use. Can be 'zero-shot' or 'one-shot'",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the output files",
    )

    args = parser.parse_args()
    if args.prompt_type not in ["zero-shot", "one-shot"]:
        raise ValueError("prompt_type must be either 'zero-shot' or 'one-shot'")

    run_gpt_on_conll(args.model, args.prompt_type, args.output_dir)
