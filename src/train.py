from model import DistilBertForClassification
from trainer import Trainer
from featurization import RottenTomatoesDataset
from torch.utils.data import DataLoader
import os
import sys
import yaml
import pickle
import torch

def main():
    params = yaml.safe_load(open("params.yaml"))["train"]

    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython train.py features model\n")
        sys.exit(1)

    input = sys.argv[1]
    output = sys.argv[2]

    #train_dataset = torch.load(os.path.join(input, "train.pt"))
    #val_dataset = torch.load(os.path.join(input, "val.pt"))
    with open(os.path.join(input, "train.pkl"), "rb") as fd:
        train_dataset = pickle.load(fd)
    with open(os.path.join(input, "val.pkl"), "rb") as fd:
        val_dataset = pickle.load(fd)

    batch_size = params["batch_size"]

    train_params = {"batch_size": batch_size,
                    "shuffle": True,
                    "num_workers": 0
                    }

    val_params = {"batch_size": batch_size,
                  "shuffle": False,
                  "num_workers": 0
                  }

    train_dataloader = DataLoader(train_dataset, **train_params)
    val_dataloader = DataLoader(val_dataset, **val_params)

    num_classes = params["num_classes"]
    dropout_rate = params["dropout_rate"]
    model_name = params["model_name"]

    config = {
        "num_classes": num_classes,
        "dropout_rate": dropout_rate
    }
    model = DistilBertForClassification(model_name, config)

    lr = params["lr"]
    n_epochs = params["n_epochs"]
    weight_decay = params["weight_decay"]

    tr_config = {
        "lr": lr,
        "n_epochs": n_epochs,
        "weight_decay": weight_decay,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    t = Trainer(tr_config)

    t.fit(
        model,
        train_dataloader,
        val_dataloader
    )

    os.makedirs("model", exist_ok=True)
    t.save(os.path.join("model", output))

if __name__ == "__main__":
    main()



