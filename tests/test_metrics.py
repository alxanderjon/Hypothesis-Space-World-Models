import torch

from hypothesis_space_world_models.evaluation.metrics import mean_squared_error


def test_mean_squared_error() -> None:
    prediction = torch.tensor([1.0, 3.0])
    target = torch.tensor([1.0, 1.0])

    assert mean_squared_error(prediction, target) == 2.0
