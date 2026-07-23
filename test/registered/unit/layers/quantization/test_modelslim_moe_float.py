"""Regression tests for ModelSlim MoE layers described as FLOAT.

ModelSlim checkpoints can contain quantized base model weights while keeping MTP
MoE expert weights unquantized. In that case the ModelSlim quantization
description marks the MoE expert projections as ``FLOAT``. The quantization
resolver should return ``None`` so SGLang falls back to the default unquantized
MoE path instead of raising while constructing speculative MTP layers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.test.test_utils import CustomTestCase


MTP_EXPERTS_LAYER = "mtp.layers.0.mlp.experts"


def _make_modelslim_moe_config(scheme_name: str):
    return {
        "quant_method": "modelslim",
        f"{MTP_EXPERTS_LAYER}.0.gate_proj.weight": scheme_name,
        f"{MTP_EXPERTS_LAYER}.0.up_proj.weight": scheme_name,
        f"{MTP_EXPERTS_LAYER}.0.down_proj.weight": scheme_name,
    }


class TestModelSlimMoEFloat(CustomTestCase):
    def test_float_moe_scheme_falls_back_to_unquantized(self):
        quant_config = ModelSlimConfig.from_config(_make_modelslim_moe_config("FLOAT"))

        scheme = quant_config.get_moe_scheme(torch.nn.Module(), MTP_EXPERTS_LAYER)

        self.assertIsNone(scheme)


if __name__ == "__main__":
    unittest.main()
