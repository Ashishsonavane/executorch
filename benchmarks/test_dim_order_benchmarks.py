# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Performance benchmarks for ExecuTorch dim order utilities.

These benchmarks cover the memory format and dimension order conversion
routines used during model export and compilation.
"""

import pytest
import torch

from executorch.exir.dim_order_utils import (
    get_dim_order,
    get_memory_format,
    is_channel_last_dim_order,
    is_contiguous_dim_order,
)


# --- get_dim_order benchmarks ---


@pytest.mark.benchmark
def test_get_dim_order_contiguous():
    """Benchmark dim order generation for contiguous format."""
    get_dim_order(torch.contiguous_format, 4)


@pytest.mark.benchmark
def test_get_dim_order_channels_last():
    """Benchmark dim order generation for channels-last format."""
    get_dim_order(torch.channels_last, 4)


@pytest.mark.benchmark
def test_get_dim_order_preserve():
    """Benchmark dim order generation for preserve format."""
    get_dim_order(torch.preserve_format, 4)


# --- get_memory_format benchmarks ---


@pytest.mark.benchmark
def test_get_memory_format_contiguous():
    """Benchmark memory format lookup for contiguous dim order."""
    get_memory_format([0, 1, 2, 3])


@pytest.mark.benchmark
def test_get_memory_format_channels_last():
    """Benchmark memory format lookup for channels-last dim order."""
    get_memory_format([0, 2, 3, 1])


@pytest.mark.benchmark
def test_get_memory_format_none():
    """Benchmark memory format lookup for None dim order."""
    get_memory_format(None)


# --- is_channel_last_dim_order benchmarks ---


@pytest.mark.benchmark
def test_is_channel_last_true():
    """Benchmark channels-last check on a channels-last tensor."""
    tensor = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
    is_channel_last_dim_order(tensor)


@pytest.mark.benchmark
def test_is_channel_last_false():
    """Benchmark channels-last check on a contiguous tensor."""
    tensor = torch.randn(1, 3, 224, 224)
    is_channel_last_dim_order(tensor)


# --- is_contiguous_dim_order benchmarks ---


@pytest.mark.benchmark
def test_is_contiguous_true():
    """Benchmark contiguous check on a contiguous tensor."""
    tensor = torch.randn(1, 3, 224, 224)
    is_contiguous_dim_order(tensor)


@pytest.mark.benchmark
def test_is_contiguous_false():
    """Benchmark contiguous check on a channels-last tensor."""
    tensor = torch.randn(1, 3, 224, 224).to(memory_format=torch.channels_last)
    is_contiguous_dim_order(tensor)
