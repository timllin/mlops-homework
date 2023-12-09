import torch
import os
import pickle
import yaml
import sys
from trainer import Trainer
from torch.utils.data import DataLoader
from sklearn import metrics
from featurization import RottenTomatoesDataset
from dvclive import Live


def config(model_file, train_file, val_file):
    params = yaml.safe_load(open("params.yaml"))["evaluate"]
    trainer = Trainer.load(model_file)


    train_params = {"batch_size": params['batch_size'],
                    "shuffle": True,
                    "num_workers": 0
                    }

    val_params = {"batch_size": params['batch_size'],
                  "shuffle": False,
                  "num_workers": 0
                  }
    with open(train_file, "rb") as fd:
        train_dataset = pickle.load(fd)
    with open(val_file, "rb") as fd:
        val_dataset = pickle.load(fd)

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **val_params)
    return trainer, train_dataloader, val_dataloader

def evaluate(trainer, loader, split, live, save_path):
    predictions = trainer.predict(loader)
    labels = []
    with torch.no_grad():
        for batch in loader:
            targets = batch['targets'].to(trainer.device, dtype=torch.long)
            labels.extend(targets)

    acc = metrics.accuracy_score(labels, predictions)
    precision = metrics.precision_score(labels, predictions, average='macro')
    recall = metrics.recall_score(labels, predictions, average='macro')

    if not live.summary:
        live.summary = {"acc": {}, "precision": {}, "recall": {}}
    live.summary["acc"][split] = acc
    live.summary["precision"][split] = precision
    live.summary["recall"][split] = recall



def main():
    EVAL_PATH = "eval"

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython evaluate.py model features\n")
        sys.exit(1)

    model_file = os.path.join("model/", sys.argv[1])
    train_file = os.path.join(sys.argv[2], "train.pkl")
    val_file = os.path.join(sys.argv[2], "val.pkl")

    trainer, train_dataloader, val_dataloader = config(model_file, train_file, val_file)

    with Live(EVAL_PATH, dvcyaml=False) as live:
        evaluate(trainer, train_dataloader, "train", live, save_path=EVAL_PATH)
        evaluate(trainer, val_dataloader, "val", live, save_path=EVAL_PATH)

if __name__ == "__main__":
    main()
