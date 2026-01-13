from src.mlops.model import Model
import torch
from hydra import initialize, compose
from tests import _PROJECT_ROOT

def test_model():
    """Test model class"""
    with initialize(config_path="../src/mlops/config", version_base="1.1"):
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

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)

    assert output.shape == (1, 10)