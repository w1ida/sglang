"""Unit tests for Qwen3.5 MTP ModelSlim draft quantization guard."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.models.qwen3_5_mtp import _has_incomplete_modelslim_mtp_moe_config
from sglang.test.test_utils import CustomTestCase


class TestQwen35MTPModelSlimGuard(CustomTestCase):
    def test_complete_mtp_moe_config_is_not_incomplete(self):
        quant_description = {
            "mtp.layers.0.mlp.experts.0.gate_proj.weight": "W8A8_DYNAMIC",
            "mtp.layers.0.mlp.experts.0.up_proj.weight": "W8A8_DYNAMIC",
            "mtp.layers.0.mlp.experts.0.down_proj.weight": "W8A8_DYNAMIC",
        }
        self.assertFalse(_has_incomplete_modelslim_mtp_moe_config(quant_description))

    def test_partial_mtp_moe_config_is_incomplete(self):
        quant_description = {
            "model.mtp.layers.0.mlp.experts.0.gate_proj.weight": "FLOAT",
            "model.mtp.layers.0.mlp.experts.0.down_proj.weight": "FLOAT",
        }
        self.assertTrue(_has_incomplete_modelslim_mtp_moe_config(quant_description))


if __name__ == "__main__":
    unittest.main()
