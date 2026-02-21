# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Performance benchmarks for ExecuTorch core tensor utilities.

These benchmarks cover the key tensor metadata operations used throughout
the ExecuTorch export and compilation pipeline.
"""

import pytest
import torch

from executorch.exir.tensor import (
    calculate_aligned_num_bytes,
    contiguous_stride_from_shape,
    dim_order_from_stride,
    num_bytes_from_shape_and_dtype,
    stride_from_dim_order,
    TensorSpec,
)


# --- num_bytes_from_shape_and_dtype benchmarks ---


@pytest.mark.benchmark
def test_num_bytes_small_tensor():
    """Benchmark byte calculation for a small 2D tensor."""
    shape = torch.Size([4, 4])
    num_bytes_from_shape_and_dtype(shape, torch.float32)


@pytest.mark.benchmark
def test_num_bytes_medium_tensor():
    """Benchmark byte calculation for a medium 4D tensor (typical CNN input)."""
    shape = torch.Size([1, 3, 224, 224])
    num_bytes_from_shape_and_dtype(shape, torch.float32)


@pytest.mark.benchmark
def test_num_bytes_large_tensor():
    """Benchmark byte calculation for a large tensor (typical LLM embedding)."""
    shape = torch.Size([1, 2048, 4096])
    num_bytes_from_shape_and_dtype(shape, torch.float16)


# --- contiguous_stride_from_shape benchmarks ---


@pytest.mark.benchmark
def test_contiguous_stride_2d():
    """Benchmark stride computation for a 2D shape."""
    shape = torch.Size([32, 64])
    contiguous_stride_from_shape(shape)


@pytest.mark.benchmark
def test_contiguous_stride_4d():
    """Benchmark stride computation for a 4D shape (CNN feature map)."""
    shape = torch.Size([1, 64, 56, 56])
    contiguous_stride_from_shape(shape)


@pytest.mark.benchmark
def test_contiguous_stride_5d():
    """Benchmark stride computation for a 5D shape (video tensor)."""
    shape = torch.Size([1, 3, 16, 224, 224])
    contiguous_stride_from_shape(shape)


# --- dim_order_from_stride benchmarks ---


@pytest.mark.benchmark
def test_dim_order_contiguous():
    """Benchmark dim order derivation from contiguous strides."""
    stride = (802816, 3136, 56, 1)
    dim_order_from_stride(stride)


@pytest.mark.benchmark
def test_dim_order_channels_last():
    """Benchmark dim order derivation from channels-last strides."""
    stride = (3136, 1, 56, 56)
    dim_order_from_stride(stride)


@pytest.mark.benchmark
def test_dim_order_5d():
    """Benchmark dim order derivation from 5D strides."""
    stride = (2408448, 802816, 50176, 224, 1)
    dim_order_from_stride(stride)


# --- stride_from_dim_order benchmarks ---


@pytest.mark.benchmark
def test_stride_from_dim_order_contiguous():
    """Benchmark stride reconstruction from contiguous dim order."""
    sizes = [1, 64, 56, 56]
    dim_order = [0, 1, 2, 3]
    stride_from_dim_order(sizes, dim_order)


@pytest.mark.benchmark
def test_stride_from_dim_order_channels_last():
    """Benchmark stride reconstruction from channels-last dim order."""
    sizes = [1, 64, 56, 56]
    dim_order = [0, 2, 3, 1]
    stride_from_dim_order(sizes, dim_order)


# --- calculate_aligned_num_bytes benchmarks ---


@pytest.mark.benchmark
def test_aligned_num_bytes():
    """Benchmark aligned byte calculation."""
    calculate_aligned_num_bytes(1024, 16)


@pytest.mark.benchmark
def test_aligned_num_bytes_unaligned():
    """Benchmark aligned byte calculation for non-aligned input."""
    calculate_aligned_num_bytes(1000, 16)


# --- TensorSpec benchmarks ---


@pytest.mark.benchmark
def test_tensor_spec_creation_small():
    """Benchmark TensorSpec creation for a small tensor."""
    TensorSpec(
        dtype=torch.float32,
        shape=torch.Size([4, 4]),
    )


@pytest.mark.benchmark
def test_tensor_spec_creation_4d():
    """Benchmark TensorSpec creation for a typical 4D CNN tensor."""
    TensorSpec(
        dtype=torch.float32,
        shape=torch.Size([1, 64, 56, 56]),
    )


@pytest.mark.benchmark
def test_tensor_spec_from_tensor():
    """Benchmark TensorSpec.from_tensor for a contiguous tensor."""
    tensor = torch.randn(1, 3, 224, 224)
    TensorSpec.from_tensor(tensor, const=False)


@pytest.mark.benchmark
def test_tensor_spec_allocated_memory():
    """Benchmark allocated_memory property access."""
    spec = TensorSpec(
        dtype=torch.float32,
        shape=torch.Size([1, 64, 56, 56]),
    )
    _ = spec.allocated_memory
