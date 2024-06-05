import json
import os
import time
import sys, glob

from vllm import LLM, SamplingParams

files = sorted(glob.glob("./n2c2/*"))
texts = []
for filename in files:
    text = open(filename).read()
    texts.append(text)
len(texts)


# Prompts

system_prompt ="You are a highly intelligent and accurate medical domain named-entity recognition (NER) system."
prompt_single_line = """In the following text, extract all substrings containing the following entity types as a JSON:
- First names of people (key "FIRST_NAME") 
- Last names of people (key "LAST_NAME")
- Middle names and initials of people (key "MIDDLE_NAME")
- Human ages (key "AGE")
- IDs of medical records (key "MED_ID")
- IDs of documents (key "DOC_ID")
- Calendar dates (key "DATE")
- Cities (key "CITY")
- States (key "STATE")
- Street addresses (key "STREET")
- Countries (key "COUNTRY")
- Zip codes (key "ZIPCODE")
- Names of hospitals and other organizations (key "ORGANIZATION")
- Email addresses (key "EMAIL")
- Phone numbers (key "PHONE") 
Return substrings such as dates in every format in which they appear in the text.
"""
zero_shot_prompt = lambda text: f"{prompt_single_line}\nInput text:\n{text}\n\nOutput JSON Response:\n"
one_shot_prompt = lambda text: f"{prompt_single_line}{one_shot_example_text}\nInput text:\n{text}\n\nOutput JSON Response:\n"

# LLAMA
all_inputs = [
    f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

{prompt_single_line}
Input text:
{text.strip()}

[/INST]

Output JSON Response:
""" 
    for text in texts]

# Mistral
all_inputs = [
    f"""<s>[INST]\n{system_prompt}\n{prompt}\n{text}\n[/INST]\n Here is the JSON object with the requested information extracted from the text:\n"""
    for text in texts
]

# Model Loading

sampling_params = SamplingParams(temperature=0, max_tokens=500)

llm = LLM(model="/data/models/Llama-2-13b-chat-hf", tensor_parallel_size=8)
start = time.time()
outputs_temp = llm.generate(all_inputs, sampling_params)
end = time.time()
print(end - start)

import json
outputs = []
for output in outputs_temp:
    generated_text = output.outputs[0].text
    outputs.append(generated_text)

with open("v2_llama_70B_results_i2b2.json", "w") as f:
    f.write(json.dumps(outputs))






