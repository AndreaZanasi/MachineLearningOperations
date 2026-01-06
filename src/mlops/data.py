from pathlib import Path

import typer
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self, data_path: Path) -> None:
        self.data_path = data_path
        self.train_set = None
        self.test_set = None

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.images)

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""
        return self.images[index], self.targets[index]

    def preprocess(self, output_folder: Path) -> None:
        """Preprocess the raw data and save it to the output folder."""
        output_folder.mkdir(parents=True, exist_ok=True)

        train_images, train_target = [], []
        for i in range(6):
            train_images.append(torch.load(f"{self.data_path}/train_images_{i}.pt"))
            train_target.append(torch.load(f"{self.data_path}/train_target_{i}.pt"))
        
        train_images = normalize(torch.cat(train_images).unsqueeze(1).float())
        train_target = torch.cat(train_target).long()

        test_images = normalize(torch.load(f"{self.data_path}/test_images.pt").unsqueeze(1).float())
        test_target = torch.load(f"{self.data_path}/test_target.pt").long()

        torch.save(train_images, f"{output_folder}/train_images.pt")
        torch.save(train_target, f"{output_folder}/train_target.pt")
        torch.save(test_images, f"{output_folder}/test_images.pt")
        torch.save(test_target, f"{output_folder}/test_target.pt")

        self.train_set = torch.utils.data.TensorDataset(train_images, train_target)
        self.test_set = torch.utils.data.TensorDataset(test_images, test_target)

        print("Data Preprocessed!")

def preprocess(data_path: Path, output_folder: Path) -> None:
    print("Preprocessing data...")
    dataset = MyDataset(data_path)
    dataset.preprocess(output_folder)

def normalize(t: torch.Tensor):
    return (t - t.mean()) / t.std()


if __name__ == "__main__":
    typer.run(preprocess)
