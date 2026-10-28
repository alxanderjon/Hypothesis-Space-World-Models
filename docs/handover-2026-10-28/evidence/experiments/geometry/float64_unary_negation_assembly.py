"""Exact zero-parameter IEEE-754 binary64 unary negation.

Unary negation toggles only the sign bit.  Bit extraction/packing is the same
deterministic scaffold already used around the locked one-bit XOR assembly;
the fixed XOR operand is the binary64 sign mask, so no learned parameters or
arithmetic reassociation are involved.  NaN payload and quiet/signaling bits
are preserved.
"""
from __future__ import annotations

import numpy as np
import torch

from experiments.sho.validate_shared_one_bit_xor import XorAssembly


FLOAT64_SIGN_MASK = np.uint64(0x8000_0000_0000_0000)


class Float64UnaryNegationAssembly:
    """Flip exactly the sign bit of each CPU float64 value."""

    new_parameters = 0

    def __init__(self) -> None:
        self.xor = XorAssembly()

    def numpy(self, values) -> np.ndarray:
        source = np.asarray(values, dtype=np.float64)
        bits = source.view(np.uint64)
        sign = ((bits >> np.uint64(63)) & np.uint64(1)).astype(np.float64)
        flipped = self.xor(sign, np.ones_like(sign))
        if not np.all((flipped == 0.0) | (flipped == 1.0)):
            raise AssertionError("locked XOR left the one-bit state space")
        output_bits = (
            (bits & ~FLOAT64_SIGN_MASK)
            | (flipped.astype(np.uint64) << np.uint64(63))
        )
        return output_bits.view(np.float64).reshape(source.shape)

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        if values.dtype != torch.float64:
            raise TypeError("unary-negation assembly requires float64")
        if values.device.type != "cpu":
            raise ValueError("the recovered production path is CPU float64")
        result = self.numpy(values.detach().numpy())
        return torch.from_numpy(result.copy()).reshape(values.shape)
