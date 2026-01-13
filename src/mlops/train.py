import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from evaluate import eval
from hydra import compose, initialize
from model import Model
from tqdm import tqdm

from data import MyDataset

log = logging.getLogger(__name__)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train(
    batch_size: int,
    epochs: int,
    data_dir: str,
    output_dir: str,
    figures_dir: str,
    model_name: str,
    model,
    optimizer,
    criterion,
):
    dataset = MyDataset(data_dir)
    dataset.preprocess(output_dir)
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size)

    statistics = {"loss": [], "accuracy": []}

    for e in tqdm(range(epochs), desc="Training"):
        model.train()
        for image, target in train_dataloader:
            image, target = image.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            prediction = model(image)
            loss = criterion(prediction, target)
            accuracy = (prediction.argmax(dim=1) == target).float().mean()
            loss.backward()

            optimizer.step()

        statistics["loss"].append(loss.item())
        statistics["accuracy"].append(accuracy.item())
        log.info(f"\nEpoch: {e} | Loss: {loss.item()} | Accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), model_name)
        eval(model_name, batch_size, dataset, model)

    log.info("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig(f"{figures_dir}/training_statistics.png")


def main():
    with initialize(config_path="config", version_base="1.1"):
        train_cfg = compose(config_name="cfg_train.yaml")
        model_cfg = compose(config_name="cfg_model.yaml")

    model = Model(
        model_cfg.hyperparameters.features,
        model_cfg.hyperparameters.batch_norms,
        model_cfg.hyperparameters.dropout,
        model_cfg.hyperparameters.stride,
        model_cfg.hyperparameters.kernel_size,
        model_cfg.max_pooling.pool_size,
        model_cfg.max_pooling.stride,
    )
    model.to(DEVICE)

    train(
        train_cfg.hyperparameters.batch_size,
        train_cfg.hyperparameters.epochs,
        train_cfg.paths.data_dir,
        train_cfg.paths.output_dir,
        train_cfg.paths.figures_dir,
        train_cfg.paths.model_name,
        model,
        hydra.utils.instantiate(train_cfg.optimizer, params=model.parameters()),
        hydra.utils.instantiate(train_cfg.criterion),
    )


if __name__ == "__main__":
    main()
