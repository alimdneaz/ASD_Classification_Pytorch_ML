import pandas as pd
import numpy as np
import torch


def load_asd_data(file_path):
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    df["label"] = df["label"].map({"ASD": 1, "Control": 0})
    X = df.drop("label", axis=1).values
    y = df["label"].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(
        y, dtype=torch.float32
    ).view(-1, 1)
