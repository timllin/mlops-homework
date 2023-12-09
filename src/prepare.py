import os
import random
import re
import sys


import yaml

def process_posts(input_lines, fd_out_train, fd_out_test, split):
    header = input_lines[0]
    input_lines = input_lines[1:]
    random.shuffle(input_lines)

    split_index = int(len(input_lines) * split)

    fd_out_train.write(header)
    fd_out_train.writelines(input_lines[:split_index])
    fd_out_test.write(header)
    fd_out_test.writelines(input_lines[split_index:])


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-file\n")
        sys.exit(1)

    # Test data set split ratio
    split = params["split"]
    random.seed(params["seed"])

    input = sys.argv[1]
    output_train = os.path.join("data", "prepared", "train.csv")
    output_test = os.path.join("data", "prepared", "val.csv")

    os.makedirs(os.path.join("data", "prepared"), exist_ok=True)

    input_lines = []
    with open(input) as fd_in:
        input_lines = fd_in.readlines()

    fd_out_train = open(output_train, "w", encoding="utf-8")
    fd_out_test = open(output_test, "w", encoding="utf-8")

    process_posts(
        input_lines=input_lines,
        fd_out_train=fd_out_train,
        fd_out_test=fd_out_test,
        split=split,
    )

    fd_out_train.close()
    fd_out_test.close()


if __name__ == "__main__":
    main()