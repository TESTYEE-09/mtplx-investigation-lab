from __future__ import annotations

import mlx.nn as nn

from mtplx.mtp_patch import MTPContract
from mtplx.step3p5_mtp_patch import _quantize_step_mtp_module


class _TinyStepLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared_head_head = nn.Linear(32, 8, bias=False)


class _TinyStepMTP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [_TinyStepLayer()]


def test_quantize_step_mtp_module_honors_contract_bits() -> None:
    mtp = _TinyStepMTP()

    _quantize_step_mtp_module(
        mtp,
        MTPContract(mtp_quant_bits=4, mtp_quant_group_size=32, mtp_quant_mode="affine"),
    )

    assert isinstance(mtp.layers[0].shared_head_head, nn.QuantizedLinear)
    assert mtp.layers[0].shared_head_head.bits == 4
    assert mtp.layers[0].shared_head_head.group_size == 32
