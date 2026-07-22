"""Unit tests for ModelSlim MoE scheme fallback behavior on MTP layers."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.layers.quantization.modelslim.modelslim import ModelSlimConfig
from sglang.srt.layers.quantization.modelslim.schemes import ModelSlimW8A8Int8MoE
from sglang.test.test_utils import CustomTestCase


class TestModelSlimMoeSchemeFallback(CustomTestCase):
    def test_complete_mtp_moe_scheme_resolves_quantized(self):
        quant_config = ModelSlimConfig(
            {
                "model.mtp.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
                "model.mtp.layers.0.mlp.experts.0.up_proj.weight": "W8A8_DYNAMIC",
                "model.mtp.layers.0.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
            }
        )
        schemes = quant_config.get_moe_scheme(
            torch.nn.Module(), "mtp.layers.0.mlp.experts"
        )
        self.assertIsNotNone(schemes)
        w13_scheme, w2_scheme = schemes
        self.assertIsInstance(w13_scheme, ModelSlimW8A8Int8MoE)
        self.assertIsInstance(w2_scheme, ModelSlimW8A8Int8MoE)

    def test_all_float_mtp_moe_scheme_falls_back(self):
        quant_config = ModelSlimConfig(
            {
                "mtp.layers.0.mlp.experts.0.gate_proj.weight": "FLOAT",
                "mtp.layers.0.mlp.experts.0.up_proj.weight": "FLOAT",
                "mtp.layers.0.mlp.experts.0.down_proj.weight": "FLOAT",
            }
        )
        schemes = quant_config.get_moe_scheme(
            torch.nn.Module(), "mtp.layers.0.mlp.experts"
        )
        self.assertIsNone(schemes)

    def test_partial_float_missing_mtp_moe_scheme_falls_back(self):
        quant_config = ModelSlimConfig(
            {
                "model.mtp.layers.0.mlp.experts.0.gate_proj.weight": "FLOAT",
                "model.mtp.layers.0.mlp.experts.0.down_proj.weight": "FLOAT",
            }
        )
        schemes = quant_config.get_moe_scheme(
            torch.nn.Module(), "mtp.layers.0.mlp.experts"
        )
        self.assertIsNone(schemes)


if __name__ == "__main__":
    unittest.main()
