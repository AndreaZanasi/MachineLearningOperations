import torch
from model import Model
from data import MyDataset
from pathlib import Path

def eval(model_checkpoint: str, device : torch.device, output_folder : Path, batch_size : int):
    
    model = Model().to(device)
    model.load_state_dict(torch.load(model_checkpoint))

    dataset = MyDataset("data/raw")
    dataset.preprocess(output_folder)
    test_dataloader = torch.utils.data.DataLoader(dataset.test_set, batch_size)

    model.eval()
    correct, total = 0, 0

    for image, target in test_dataloader:
        image, target = image.to(device), target.to(device)
        prediction = model(image)
        correct += (prediction.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)

    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    eval("models/model.pth", DEVICE, Path("data/processed"), 64)