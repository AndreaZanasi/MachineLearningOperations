import torch
import torch.nn.functional as F
from torch import nn
import hydra

class Model(nn.Module):
    """My awesome model."""

    def __init__(
            self,
            features,
            batch_norms,
            dropout,
            stride,
            kernel_size,
            max_pool_size,
            max_pool_stride
        ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features[0], features[1], kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(batch_norms[0])
        self.conv2 = nn.Conv2d(features[1], features[2], kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(batch_norms[1])
        self.conv3 = nn.Conv2d(features[2], features[3], kernel_size, stride)
        self.bn3 = nn.BatchNorm2d(batch_norms[2])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(features[3], features[4])

        self.max_pool_size = max_pool_size
        self.max_pool_stride = max_pool_stride

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, self.max_pool_size, self.max_pool_stride)
        x = F.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, self.max_pool_size, self.max_pool_stride)
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, self.max_pool_size, self.max_pool_stride)
        x = self.dropout(torch.flatten(x, 1))

        return self.fc1(x)

@hydra.main(config_path="config", config_name="cfg_model", version_base="1.1")
def main(model_cfg):
    model = Model(
        model_cfg.hyperparameters.features,
        model_cfg.hyperparameters.batch_norms,
        model_cfg.hyperparameters.dropout,
        model_cfg.hyperparameters.stride,
        model_cfg.hyperparameters.kernel_size,
        model_cfg.max_pooling.pool_size,
        model_cfg.max_pooling.stride,
    )

    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
