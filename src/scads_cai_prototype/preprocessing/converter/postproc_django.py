import argparse

from scads_cai_prototype.preprocessing.filehandling import load_json, save_json


def postprocess_predictions(predictions, replacements):
    pred_with_repl = predictions.merge(replacements, on='id')

    replaced_expected = pred_with_repl.apply(lambda row: replace(row['expected_codesnippet'], row['replacements']),
                                             axis=1)
    pred_with_repl['expected_codesnippet'] = replaced_expected

    replaced_predicted = pred_with_repl.apply(
        lambda row: [replace(predicted, row['replacements']) for predicted in row['predicted_codesnippets']], axis=1)
    pred_with_repl['predicted_codesnippets'] = replaced_predicted

    return pred_with_repl[['expected_codesnippet', 'predicted_codesnippets', 'id']]


def replace(code, replacements):
    for name, value in replacements.items():
        code = code.replace("'" + name + "'", value)

    return code


def load_replacements(dataset_input_file):
    dataset = load_json(dataset_input_file)
    return dataset[["id", "replacements"]]


def get_args():
    parser = argparse.ArgumentParser("Evaluate predicted target file for Django", fromfile_prefix_chars='@')

    parser.add_argument("--predictions-input-file", type=str, default="django-predicted.jsonl")
    parser.add_argument("--dataset-input-file", type=str, default="django-test.jsonl")
    parser.add_argument("--predictions-output-file", type=str, default="django-predicted-postproc.jsonl")

    return parser.parse_args()


def main():
    args = get_args()
    print("Django postproc args:", vars(args))

    predictions = load_json(args.predictions_input_file)
    replacements = load_replacements(args.dataset_input_file)

    postprocessed_predictions = postprocess_predictions(predictions, replacements)
    save_json(postprocessed_predictions, args.predictions_output_file)


if __name__ == '__main__':
    main()
