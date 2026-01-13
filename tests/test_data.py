import os.path

import pytest
import torch
from torch.utils.data import Dataset

from src.mlops.data import MyDataset


@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    dataset = MyDataset("data/raw")
    assert isinstance(dataset, Dataset)

    dataset.preprocess("data/processed")
    assert len(dataset.train_set) == 30000
    assert len(dataset.test_set) == 5000

    for ds in [dataset.train_set, dataset.test_set]:
        for image, label in ds:
            assert image.shape == (1, 28, 28)
            assert label in range(10)

    assert (torch.unique(dataset.train_set.tensors[1]) == torch.arange(0, 10)).all()
    assert (torch.unique(dataset.test_set.tensors[1]) == torch.arange(0, 10)).all()
