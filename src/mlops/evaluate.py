from pathlib import Path

import hydra
import torch
from hydra import compose, initialize
from model import Model

from data import MyDataset

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def eval(
    model_checkpoint: str,
    batch_size: int,
    dataset: torch.utils.data.TensorDataset,
    model: Model,
):
    model.load_state_dict(torch.load(model_checkpoint))
    test_dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for image, target in test_dataloader:
            image, target = image.to(DEVICE), target.to(DEVICE)
            prediction = model(image)
            correct += (prediction.argmax(dim=1) == target).float().sum().item()
            total += target.size(0)

    print(f"Test accuracy: {correct / total}")


def main():
    with initialize(config_path="config", version_base="1.1"):
        train_cfg = compose(config_name="cfg_train.yaml")
        model_cfg = compose(config_name="cfg_model.yaml")

    dataset = MyDataset(train_cfg.paths.data_dir)
    dataset.preprocess(train_cfg.paths.output_dir)

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

    eval(
        train_cfg.paths.model_name, train_cfg.hyperparameters.batch_size, dataset, model
    )


if __name__ == "__main__":
    main()
