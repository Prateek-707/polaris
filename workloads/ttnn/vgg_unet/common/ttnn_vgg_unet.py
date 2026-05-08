#!/usr/bin/env python
# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Polaris-side VGG UNet model — structurally aligned with the tt-metal canonical
``models/demos/vision/segmentation/vgg_unet/common/ttnn/ttnn_vgg_unet.py``.

Differences from the canonical (all deliberate for polaris analytical modeling):
- No ``ttnn.permute`` / ``ttnn.reshape`` before the first conv: the polaris
  ``conv_sinf`` shape-inference engine expects NCHW tensors; hardware-specific
  channel-padding and memory reformatting are not modeled here.
- ``sharded_concat`` uses ``dim=1`` (NCHW channel axis) rather than the canonical
  ``dim=3`` (NHWC channel axis).  Polaris tensors are NCHW throughout because
  the permute+reshape that converts to NHWC is skipped (see above).
- ``ttnn.conv2d`` / ``ttnn.conv_transpose2d`` return a single ``Tensor`` in the
  polaris shim (no list unpacking of output_dim / weight cache).
- ``ttnn.sigmoid`` at the end uses no shard-spec introspection (the polaris Tensor
  does not carry a live memory config).
"""

import ttsim.front.ttnn as ttnn


# ---------------------------------------------------------------------------
# sharded_concat — canonical HEIGHT-shard pattern, list-first calling convention
# ---------------------------------------------------------------------------

def sharded_concat(input_tensors, num_cores=64, dim=1):
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = ((input_tensors[0].shape[2]) + num_cores - 1) // num_cores
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)


# ---------------------------------------------------------------------------
# Conv
# ---------------------------------------------------------------------------

class Conv:
    def __init__(
        self,
        device,
        conv_param,
        conv_pth,
        input_tensor_layout=ttnn.TILE_LAYOUT,
    ) -> None:
        self.conv_param = conv_param
        self.conv_pth = conv_pth
        self.device = device
        self.cache = {}  # type: ignore[var-annotated]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        self.conv_output_dtype = conv_param.dtype
        output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=conv_param.activation,
            shard_layout=conv_param.shard_layout,
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
            deallocate_activation=conv_param.deallocate_activation,
            enable_act_double_buffer=conv_param.enable_act_double_buffer,
            enable_weights_double_buffer=True,
            output_layout=output_layout,
        )
        if conv_param.act_block_h is not None:
            self.conv_config.act_block_h_override = conv_param.act_block_h

        self.bias = conv_pth.bias
        self.weight = conv_pth.weight

        self.conv_kwargs = {
            'in_channels': conv_param.in_channels,
            'out_channels': conv_param.out_channels,
            'batch_size': conv_param.batch_size,
            'input_height': conv_param.input_height,
            'input_width': conv_param.input_width,
            'kernel_size': conv_param.kernel_size,
            'stride': conv_param.stride,
            'padding': conv_param.padding,
            'dilation': conv_param.dilation,
            'groups': conv_param.groups,
            'device': device,
            'conv_config': self.conv_config,
        }

    def __str__(self) -> str:
        return f'Conv: {self.weight.shape} {self.bias.shape} {self.conv_kwargs["kernel_size"]}'

    def __call__(self, x):
        # Polaris shim returns a single Tensor; canonical HW form unpacks 3-element list.
        x = ttnn.conv2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **self.conv_kwargs,
            compute_config=self.compute_config,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=self.conv_output_dtype,
        )
        return x


# ---------------------------------------------------------------------------
# Conv_transpose
# ---------------------------------------------------------------------------

class Conv_transpose:
    def __init__(
        self,
        device,
        conv_param,
        conv_pth,
    ) -> None:
        self.conv_param = conv_param
        self.conv_pth = conv_pth
        self.device = device
        self.cache = {}  # type: ignore[var-annotated]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
            math_approx_mode=True,
        )
        output_layout = ttnn.ROW_MAJOR_LAYOUT

        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            shard_layout=conv_param.shard_layout,
            reshard_if_not_optimal=conv_param.reshard_if_not_optimal,
            deallocate_activation=conv_param.deallocate_activation,
            enable_act_double_buffer=conv_param.enable_act_double_buffer,
            enable_weights_double_buffer=True,
            output_layout=output_layout,
        )
        if conv_param.act_block_h is not None:
            self.conv_config.act_block_h_override = conv_param.act_block_h

        self.bias = conv_pth.bias
        self.weight = conv_pth.weight

        self.conv_kwargs = {
            'in_channels': conv_param.in_channels,
            'out_channels': conv_param.out_channels,
            'batch_size': conv_param.batch_size,
            'input_height': conv_param.input_height,
            'input_width': conv_param.input_width,
            'kernel_size': conv_param.kernel_size,
            'stride': conv_param.stride,
            'padding': conv_param.padding,
            'dilation': conv_param.dilation,
            'groups': conv_param.groups,
            'device': device,
            'conv_config': self.conv_config,
            'output_padding': conv_param.output_padding,
        }

    def __str__(self) -> str:
        return f'Conv_transpose: {self.weight.shape} {self.bias.shape} {self.conv_kwargs["kernel_size"]}'

    def __call__(self, x):
        # Polaris shim returns a single Tensor; canonical HW form unpacks [x, [w, b]].
        x = ttnn.conv_transpose2d(
            input_tensor=x,
            weight_tensor=self.weight,
            bias_tensor=self.bias,
            **self.conv_kwargs,
            compute_config=self.compute_config,
            return_weights_and_bias=True,
            mirror_kernel=True,
            dtype=self.conv_param.dtype,
        )
        return x


# ---------------------------------------------------------------------------
# Tt_decoder_block
# ---------------------------------------------------------------------------

class Tt_decoder_block:
    def __init__(self, device, conv_args, parameters) -> None:
        self.conv_args = conv_args
        self.up = Conv_transpose(device, conv_args.up, parameters.up)
        self.conv1 = Conv(device, conv_args.conv_block.conv1, parameters.conv1)
        self.conv2 = Conv(device, conv_args.conv_block.conv2, parameters.conv2)

    def __call__(self, x, cat_in):
        x = self.up(x)
        x = sharded_concat([x, cat_in])
        if self.conv_args.conv_block.conv1.do_sharded_to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.conv1(x)
        if self.conv_args.conv_block.conv2.do_sharded_to_interleaved:
            x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.conv2(x)
        return x


# ---------------------------------------------------------------------------
# Tt_vgg_unet
# ---------------------------------------------------------------------------

class Tt_vgg_unet:
    def __init__(self, device, parameters, conv_args) -> None:
        self.conv_args = conv_args
        self.parameters = parameters
        self.s1_0 = Conv(device, conv_args.s1['0'], parameters['0'], input_tensor_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.s1_2 = Conv(device, conv_args.s1['2'], parameters['2'])
        self.s2_5 = Conv(device, conv_args.s2['5'], parameters['5'])
        self.s2_7 = Conv(device, conv_args.s2['7'], parameters['7'])
        self.s3_10 = Conv(device, conv_args.s3['10'], parameters['10'])
        self.s3_12 = Conv(device, conv_args.s3['12'], parameters['12'])
        self.s3_14 = Conv(device, conv_args.s3['14'], parameters['14'])
        self.s3_16 = Conv(device, conv_args.s3['16'], parameters['16'])
        self.s4_19 = Conv(device, conv_args.s4['19'], parameters['19'])
        self.s4_21 = Conv(device, conv_args.s4['21'], parameters['21'])
        self.s4_23 = Conv(device, conv_args.s4['23'], parameters['23'])
        self.s4_25 = Conv(device, conv_args.s4['25'], parameters['25'])
        self.b1_28 = Conv(device, conv_args.b1['28'], parameters['28'])
        self.b1_30 = Conv(device, conv_args.b1['30'], parameters['30'])
        self.b1_32 = Conv(device, conv_args.b1['32'], parameters['32'])
        self.b1_34 = Conv(device, conv_args.b1['34'], parameters['34'])
        self.d1 = Tt_decoder_block(device, conv_args.d1, parameters.d1)
        self.d2 = Tt_decoder_block(device, conv_args.d2, parameters.d2)
        self.d3 = Tt_decoder_block(device, conv_args.d3, parameters.d3)
        self.d4 = Tt_decoder_block(device, conv_args.d4, parameters.d4)
        self.out = Conv(device, conv_args.out, parameters.out)

    def __call__(self, input):
        # Polaris models the mathematical computation directly on NCHW input.
        # The canonical HW form pads channels to 16, then permutes+reshapes to
        # [1, 1, N*H*W, C] format — those hardware-specific ops are skipped here
        # because conv_sinf expects NCHW and the polaris shim returns a single
        # Tensor from conv2d (not the unpacked [out, dims, weights] tuple).
        x = input

        x = self.s1_0(x)
        x = self.s1_2(x)
        s1 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s2['4'].batch_size,
            input_h=self.conv_args.s2['4'].input_height,
            input_w=self.conv_args.s2['4'].input_width,
            channels=x.shape[1],
            kernel_size=[self.conv_args.s2['4'].kernel_size, self.conv_args.s2['4'].kernel_size],
            stride=[self.conv_args.s2['4'].stride, self.conv_args.s2['4'].stride],
            padding=[self.conv_args.s2['4'].padding, self.conv_args.s2['4'].padding],
            dilation=[self.conv_args.s2['4'].dilation, self.conv_args.s2['4'].dilation],
        )
        x = self.s2_5(x)
        x = self.s2_7(x)
        s2 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s3['9'].batch_size,
            input_h=self.conv_args.s3['9'].input_height,
            input_w=self.conv_args.s3['9'].input_width,
            channels=x.shape[1],
            kernel_size=[self.conv_args.s3['9'].kernel_size, self.conv_args.s3['9'].kernel_size],
            stride=[self.conv_args.s3['9'].stride, self.conv_args.s3['9'].stride],
            padding=[self.conv_args.s3['9'].padding, self.conv_args.s3['9'].padding],
            dilation=[self.conv_args.s3['9'].dilation, self.conv_args.s3['9'].dilation],
        )

        x = self.s3_10(x)

        shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
        sharded_memory_config = ttnn.create_sharded_memory_config(
            [512, 32],
            core_grid=shard_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            use_height_and_width_as_shard_shape=True,
        )
        x = ttnn.to_memory_config(x, sharded_memory_config)

        x = self.s3_12(x)
        x = self.s3_14(x)
        x = self.s3_16(x)
        s3 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.s4['18'].batch_size,
            input_h=self.conv_args.s4['18'].input_height,
            input_w=self.conv_args.s4['18'].input_width,
            channels=x.shape[1],
            kernel_size=[self.conv_args.s4['18'].kernel_size, self.conv_args.s4['18'].kernel_size],
            stride=[self.conv_args.s4['18'].stride, self.conv_args.s4['18'].stride],
            padding=[self.conv_args.s4['18'].padding, self.conv_args.s4['18'].padding],
            dilation=[self.conv_args.s4['18'].dilation, self.conv_args.s4['18'].dilation],
        )

        x = self.s4_19(x)
        x = self.s4_21(x)
        x = self.s4_23(x)
        x = self.s4_25(x)
        s4 = x

        x = ttnn.max_pool2d(
            input_tensor=x,
            batch_size=self.conv_args.b1['27'].batch_size,
            input_h=self.conv_args.b1['27'].input_height,
            input_w=self.conv_args.b1['27'].input_width,
            channels=x.shape[1],
            kernel_size=[self.conv_args.b1['27'].kernel_size, self.conv_args.b1['27'].kernel_size],
            stride=[self.conv_args.b1['27'].stride, self.conv_args.b1['27'].stride],
            padding=[self.conv_args.b1['27'].padding, self.conv_args.b1['27'].padding],
            dilation=[self.conv_args.b1['27'].dilation, self.conv_args.b1['27'].dilation],
        )

        x = self.b1_28(x)
        x = self.b1_30(x)
        x = self.b1_32(x)
        x = self.b1_34(x)

        x = self.d1(x, s4)
        ttnn.deallocate(s4)
        x = self.d2(x, s3)
        ttnn.deallocate(s3)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.d3(x, s2)
        ttnn.deallocate(s2)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = self.d4(x, s1)
        ttnn.deallocate(s1)
        x = self.out(x)

        x = ttnn.sigmoid(x)
        return x
