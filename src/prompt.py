# Prompts

system_prompt = "You are a highly intelligent and accurate medical domain named-entity recognition (NER) system."
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
one_shot_example_text = open("one_shot_example.txt").read()
# GPT
zero_shot_prompt = (
    lambda text: f"{prompt_single_line}\nInput text:\n{text}\n\nOutput JSON Response:\n"
)
one_shot_prompt = (
    lambda text: f"{prompt_single_line}{one_shot_example_text}\nInput text:\n{text}\n\nOutput JSON Response:\n"
)

# Llama-3
llama3_prompt = (
    lambda text: f"""
    <|begin_of_text|>
    <|start_header_id|>user<|end_header_id|>
    {system_prompt}
    <instruction>
    Input text:
    {text}
    <|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>

    Output JSON Response:
    """
)

# Mixtral
mixtral_prompt = (
    lambda text: f"""
    <s>[INST]
    {system_prompt}
    {prompt_single_line}
    Input text:
    {text}
    [/INST]
    Here is the JSON object with the requested information extracted from the text:
    """
)
