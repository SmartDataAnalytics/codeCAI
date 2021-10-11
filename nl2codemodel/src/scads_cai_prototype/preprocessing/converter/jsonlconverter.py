import argparse

from scads_cai_prototype.preprocessing.filehandling import load_json, save_json


def get_args():
    parser = argparse.ArgumentParser("Convert between JSON and JSONL", fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument("--input-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)

    return parser.parse_args()


def main():
    args = get_args()
    data = load_json(args.input_path)
    save_json(data, args.output_path)


if __name__ == '__main__':
    main()
