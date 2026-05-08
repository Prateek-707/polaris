#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side equivalent of vgg_unet_test_infra.py from tt-metal (WH variant).

Provides ``create_test_infra`` with the same interface so that
``test_vgg_unet_device_perf_wh.py`` can run in dual-mode.
"""

import ttsim.front.ttnn as ttnn
import ttsim.front.ttnn.minitorch_shim as torch  # type: ignore[no-redef]

from ttsim.front.ttnn.device import set_default_device
from ttsim.front.ttnn.tensor import ttnn_random
from ttsim.front.ttnn.buffer import BufferType, TensorMemoryLayout
from ttsim.front.ttnn.memory import MemoryConfig

from workloads.ttnn.vgg_unet.common.model_preprocessing import create_vgg_unet_model_parameters
from workloads.ttnn.vgg_unet.common.ttnn_vgg_unet import Tt_vgg_unet


class VggUnetTestInfra:
    """Polaris mirror of the upstream VggUnetTestInfra.

    Builds synthetic parameters and a random [B, 3, 256, 256] input tensor.
    The model is the polaris-native ``Tt_vgg_unet`` which expects NCHW input
    and returns a segmentation mask of shape [B, 1, 256, 256].
    """

    def __init__(
        self,
        device,
        batch_size,
        inputs_mesh_mapper=None,
        weights_mesh_mapper=None,
        output_mesh_composer=None,
        use_random_input_tensor: bool = False,
        model_location_generator=None,
    ):
        torch.manual_seed(0)
        set_default_device(device)

        self.device = device
        self.batch_size = batch_size

        parameters = create_vgg_unet_model_parameters(device)
        self.model = Tt_vgg_unet(device, parameters, parameters.conv_args)

        self.torch_input = ttnn_random(
            (batch_size, 3, 256, 256), -1, 1, dtype=torch.bfloat16,
        )
        self.input_tensor = None

    def setup_l1_sharded_input(self, device, torch_input=None, mesh_mapper=None, mesh_composer=None):
        if torch_input is None:
            torch_input = self.torch_input
        tt_inputs_host = ttnn.from_torch(
            torch_input, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        input_mem_config = MemoryConfig(TensorMemoryLayout.HEIGHT_SHARDED, BufferType.L1)
        return tt_inputs_host, input_mem_config

    def setup_dram_sharded_input(self, device, _torch_input_tensor=None, mesh_mapper=None, mesh_composer=None):
        tt_inputs_host, input_mem_config = self.setup_l1_sharded_input(
            device, mesh_mapper=mesh_mapper, mesh_composer=mesh_composer,
        )
        sharded_mem_config_DRAM = MemoryConfig.DRAM
        return tt_inputs_host, sharded_mem_config_DRAM, input_mem_config

    def run(self, tt_input_tensor=None):
        self.output_tensor = None
        input_tensor = tt_input_tensor if tt_input_tensor is not None else self.input_tensor
        self.output_tensor = self.model(input_tensor)
        return self.output_tensor


def create_test_infra(
    device,
    batch_size,
    inputs_mesh_mapper=None,
    weights_mesh_mapper=None,
    output_mesh_composer=None,
    use_random_input_tensor: bool = False,
    model_location_generator=None,
):
    return VggUnetTestInfra(
        device,
        batch_size,
        inputs_mesh_mapper,
        weights_mesh_mapper,
        output_mesh_composer,
        use_random_input_tensor,
        model_location_generator=model_location_generator,
    )
