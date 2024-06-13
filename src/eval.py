import argparse
import json
import os

from load_data import load_conll_data, load_i2b2_data

VALID_ENTITIES = [
    "CITY",
    "COUNTRY",
    "FIRST_NAME",
    "LAST_NAME",
    "MIDDLE_NAME",
    "ORGANIZATION",
    "ORGANISATION",
    "HOSPITAL",
    "ZIP",
    "STATE",
    "PERSON" "STREET",
    "COMPANY",
    "ZIPCODE",
    "LOCATION",
    "NAME",
]


def get_json_object(result, model):
    """
    Extract the JSON object from the result string.
    Sometimes the JSON will not be directly parsable due to the presence of additional text.
    This function will attempt to remove the extra text and return JSON.
    """
    if result.index("{") > result.index("["):
        end_idx = result.rindex("]")
        flag = True
    else:
        end_idx = result.index("}")
        flag = False
    result = result[: end_idx + 1]
    result = result.replace("```json", "").replace("```", "")
    if "mixtral" in model:
        result = result.replace("\\", "").replace("```json", "")
    doc_res = json.loads(result)
    if flag:
        doc_res = doc_res[0]
    return doc_res


def compute_precision_recall(
    results, document_labels, model, dataset, filename
) -> None:
    """
    Compute precision, recall, and F1-score based on the results and document labels.
    """
    tp = 0
    fp = 0
    fn = 0
    error = 0
    fn_list = []

    for result, ground_truth in zip(results, document_labels):

        try:
            doc_res = get_json_object(result, model)

            all_vals = set()
            for entity, vals in doc_res.items():
                if entity in VALID_ENTITIES:
                    if type(vals) is list:
                        for val in vals:
                            all_vals.update(val.lower().split())
                    else:
                        # Invalid format - Skip this entity type
                        pass

            if dataset == "conll":
                # For the conll dataset, we exclude the MISC tags from FN and TP computations
                for sentence in ground_truth:
                    for token, tag in zip(sentence["tokens"], sentence["tags"]):
                        if token.lower() not in all_vals:
                            if tag not in [0, 2, 7]:
                                fn_list.append((token, tag))
                                fn += 1
                        else:
                            if tag not in [0, 2, 7]:
                                tp += 1
                            else:
                                fp += 1
            elif dataset == "i2b2":
                gt_tags_set = set()
                for tag in ground_truth:
                    # 3rd element in the tuple is the the PII word
                    gt_tags_set.update(tag[2].lower().split())

                for val in all_vals:
                    if val in gt_tags_set:
                        tp += 1
                    else:
                        fp += 1
                for val in gt_tags_set:
                    if val not in all_vals:
                        fn += 1
        except:
            error += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1_score = 2 * recall * precision / (recall + precision)

    print("Evaluation Results:")
    print(
        f"\n\nFilename = {filename}\n\nTrue Positives: {tp}\nFalse Negatives: {fn}\nFalse Positives: {fp}\n Num errors: {error}"
    )
    print(f"\n\nPrecision = {precision}\nRecall = {recall}\nF1-Score = {f1_score}")


def evaluate(filename):
    """
    Evaluate the model performance for the provided results file.
    """
    results = json.loads(open(filename).read())
    "results_{get_current_date_as_string()}_{model}_zero-shot_{dataset}_results.json"
    _, _, model, _, dataset = os.path.basename(filename).split(".")[0].split("_")
    if dataset == "conll":
        document_labels = load_conll_data()
    else:
        _, document_labels = load_i2b2_data(dataset, return_labels=True)
    compute_precision_recall(results, document_labels, model, dataset, filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Path to the JSON file containing results",
    )

    args = parser.parse_args()
    evaluate(args.filename)
