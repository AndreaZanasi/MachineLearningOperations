from model import Model
from data import MyDataset
from tqdm import tqdm
from pathlib import Path
from evaluate import eval
import torch
import matplotlib.pyplot as plt


def train(batch_size: int, epochs: int, lr: float, device: torch.device):
    dataset = MyDataset("data/raw")
    dataset.preprocess(Path("data/processed"))
    train_dataloader = torch.utils.data.DataLoader(dataset.train_set, batch_size)

    model = Model()
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"loss": [], "accuracy": []}

    for e in tqdm(range(epochs), desc="Training"):
        model.train()
        for image, target in train_dataloader:
            image, target = image.to(device), target.to(device)

            optimizer.zero_grad()

            prediction = model(image)
            loss = criterion(prediction, target)
            accuracy = (prediction.argmax(dim=1) == target).float().mean()
            loss.backward()

            optimizer.step()

        statistics["loss"].append(loss.item())
        statistics["accuracy"].append(accuracy.item())
        print(f"\nEpoch: {e} | Loss: {loss.item()} | Accuracy: {accuracy.item()}")

        torch.save(model.state_dict(), "models/model.pth")
        eval("models/model.pth", device, batch_size, dataset)

    print("Training complete")

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["accuracy"])
    axs[1].set_title("Train accuracy")
    fig.savefig("reports/figures/training_statistics.png")


if __name__ == "__main__":
    BATCH_SIZE = 64
    EPOCHS = 30
    LR = 0.001
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    train(BATCH_SIZE, EPOCHS, LR, DEVICE)
