import torch

from hypothesis_space_world_models.models import SHOExactSolutionDecoder


def test_exact_decoder_uses_sho_solution_family() -> None:
    decoder = SHOExactSolutionDecoder(omega=2.0)
    coefficients = torch.tensor([[1.0, 0.5]])
    times = torch.tensor([0.0, torch.pi / 4.0])

    trajectories = decoder(coefficients, times)

    assert torch.allclose(trajectories, torch.tensor([[1.0, 0.5]]), atol=1e-6)
