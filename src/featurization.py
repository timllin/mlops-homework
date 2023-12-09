import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import os
import sys
import yaml
import pickle

class RottenTomatoesDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_len):
        self.data = dataframe
        self.text = dataframe['movie_description']
        self.targets = None
        if 'target' in dataframe:
            self.targets = dataframe['target']
        self.tokenizer = DistilBertTokenizer.from_pretrained(
    tokenizer, truncation=True, do_lower_case=True)
        self.max_seq_len = max_seq_len

    def __getitem__(self, index):
        text = str(self.text[index])
        text = ' '.join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        if self.targets is not None:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(self.targets[index], dtype=torch.long)
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
            }

    def __len__(self) -> int:
        return len(self.text)

def create_dataset(dataframe, tokenizer, max_seq_len):
    dataset = RottenTomatoesDataset(dataframe, tokenizer, max_seq_len)
    return dataset

def create_dataloader(input, output, tokenizer, max_seq_len):
    dataframe = pd.read_csv(input)
    dataset = create_dataset(dataframe, tokenizer, max_seq_len)

    #torch.save(dataset, output)
    with open(output, "wb") as fd:
        pickle.dump(dataset, fd)



def main():
    params = yaml.safe_load(open("params.yaml"))["featurize"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython featurization.py data-dir-path features-dir-path\n")
        sys.exit(1)

    in_path = sys.argv[1]
    out_path = sys.argv[2]

    train_input = os.path.join(in_path, "train.csv")
    val_input = os.path.join(in_path, "val.csv")

    train_output = os.path.join(out_path, "train.pkl")
    val_output = os.path.join(out_path, "val.pkl")


    tokenizer = params["tokenizer"]
    max_seq_len = params["max_seq_len"]


    os.makedirs(out_path, exist_ok=True)

    create_dataloader(train_input, train_output, tokenizer, max_seq_len)
    create_dataloader(val_input, val_output, tokenizer, max_seq_len)

if __name__ == "__main__":
    main()
