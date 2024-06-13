import glob
import json
import os
import xml.etree.ElementTree as ET

#### Functions to load CONLL 2003 data from Hugging Face ####


def download_rows():
    """
    Download the rows from the Hugging Face server in batches of 100 and combine them.
    """
    # Base URL and parameters for the curl command
    base_url = "https://datasets-server.huggingface.co/rows?dataset=tner%2Fconll2003&config=conll2003&split=test"
    # List to store the combined results of all 'rows'
    combined_rows = []

    # Loop to increment the offset from 0 to 3400 in increments of 100
    for offset in range(0, 3500, 100):
        # Constructing the complete URL with the current offset
        url = f"{base_url}&offset={offset}&length=100"

        # Executing the curl command and capturing the output
        curl_command = f'curl -X GET "{url}"'
        result = os.popen(curl_command).read()

        # Parsing the JSON result
        data = json.loads(result)

        # Appending the 'rows' from the current result to the combined list
        combined_rows.extend(data["rows"])
    return combined_rows


def is_date(token):
    """Check if the token is a date in the format yyyy-mm-dd."""
    try:
        # Trying to parse the token as a date
        year, month, day = map(int, token.split("-"))
        return 1 <= month <= 12 and 1 <= day <= 31 and len(str(year)) == 4
    except ValueError:
        # Token is not a date
        return False


def is_all_upper(tokens):
    """Check if all tokens in the list are uppercase."""
    return all(token.upper() == token for token in tokens)


def construct_documents(combined_rows):
    """
    Construct documents from the combined rows based on the presence of dates.
    """
    documents = []
    document_start_prev = 0
    document_start = None

    for i, row in enumerate(combined_rows):
        # Extracting tokens from the row
        tokens = row["row"]["tokens"]

        # Check if the last word in the sentence is a date
        if tokens and is_date(tokens[-1]):
            # Finding the previous row for which all tokens are upper case
            for j in range(i - 1, -1, -1):
                # print(j, combined_rows[j]["row"]["tokens"])
                if is_all_upper(combined_rows[j]["row"]["tokens"]):
                    document_start = j
                    break

            # print(document_start, document_start_prev)

        # Adding the current document to documents list
        if document_start is not None and document_start != document_start_prev:
            if document_start_prev is None:
                document_start_prev = 0
            documents.append(
                [r["row"] for r in combined_rows[document_start_prev:document_start]]
            )
            #         for row in combined_rows[document_start_prev:document_start]:
            #             print(row["row"]["tokens"])
            document_start_prev = document_start

    documents.append([r["row"] for r in combined_rows[document_start_prev:]])
    return documents


def load_conll_data():
    combined_rows = download_rows()
    documents = construct_documents(combined_rows)
    return documents


#### Functions to load I2B2 data from folder path ####


def cleanup_tag(tag):
    if tag.attrib["TYPE"] == "UNTAGGED":
        return None
    if tag.attrib["text"].endswith("'s"):
        tag.attrib["text"][:-2]
        tag.attrib["end"] = int(tag.attrib["end"]) - 2
    return (
        tag.attrib["start"],
        tag.attrib["end"],
        tag.attrib["text"],
        tag.attrib["TYPE"],
    )


def parse_file(i2b2_filename):
    tree = ET.parse(i2b2_filename)
    text, tags_raw = tree.getroot()[0].text, tree.getroot()[1]
    return text, tags_raw


def load_i2b2_data(i2b2_dir_path, return_labels=False):
    files = sorted(glob.glob(os.path.join(i2b2_dir_path, "*.xml")))
    texts = []
    all_tags = []
    for filename in files:
        i2b2_doc, tags_raw = parse_file(filename)
        text = i2b2_doc.text
        tags = []
        for tag in tags_raw:
            phi_tag = cleanup_tag(tag)
            if phi_tag:
                tags.append(phi_tag)
        texts.append(text)
        all_tags.append(tags)
    if return_labels:
        return texts, all_tags
    else:
        return texts
