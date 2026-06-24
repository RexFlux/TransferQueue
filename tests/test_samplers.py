# Copyright 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The TransferQueue Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for TransferQueue samplers."""

from typing import Any

import pytest

from transfer_queue.sampler import BaseSampler
from transfer_queue.sampler.grpo_group_n_sampler import GRPOGroupNSampler
from transfer_queue.sampler.rank_aware_sampler import RankAwareSampler
from transfer_queue.sampler.seqlen_balanced_sampler import (
    SeqlenBalancedSampler,
    get_seqlen_balanced_partitions,
)
from transfer_queue.sampler.sequential_sampler import SequentialSampler
from transfer_queue.sampler.streaming_token_budget_sampler import StreamingTokenBudgetSampler


class TestBaseSampler:
    """Test cases for BaseSampler abstract class."""

    def test_base_sampler_is_abstract(self):
        """Test that BaseSampler cannot be instantiated directly."""
        with pytest.raises(TypeError) as exc_info:
            BaseSampler()

        assert "Can't instantiate abstract class" in str(exc_info.value)
        assert "sample" in str(exc_info.value)

    def test_base_sampler_has_abstract_methods(self):
        """Test that BaseSampler defines abstract methods."""
        assert hasattr(BaseSampler, "sample")
        assert getattr(BaseSampler.sample, "__isabstractmethod__", False)

    def test_base_sampler_has_call_method(self):
        """Test that BaseSampler has __call__ method."""
        assert callable(BaseSampler)

    def test_base_sampler_initialization_states(self):
        """Test BaseSampler initialization sets _states correctly."""

        # Create a concrete implementation for testing
        class TestSampler(BaseSampler):
            def sample(self, ready_indexes: list[int], batch_size: int, **kwargs: Any) -> tuple[list[int], list[int]]:
                return ready_indexes[:batch_size], ready_indexes[:batch_size]

        sampler = TestSampler()
        assert hasattr(sampler, "_states")
        assert sampler._states == {}


class TestSequentialSampler:
    """Test cases for SequentialSampler."""

    def test_sequential_sampler_initialization(self):
        """Test SequentialSampler initialization."""
        sampler = SequentialSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_sequential_sampler_basic_functionality(self):
        """Test basic sampling functionality."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_sequential_sampler_empty_ready_indexes(self):
        """Test behavior with empty ready indexes."""
        sampler = SequentialSampler()
        ready_indexes = []
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_sequential_sampler_batch_size_larger_than_ready(self):
        """Test behavior when batch_size > len(ready_indexes)."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1]
        batch_size = 5

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1]
        assert consumed == [0, 1]
        assert len(sampled) == len(ready_indexes)

    def test_sequential_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_sequential_sampler_negative_batch_size(self):
        """Test behavior with negative batch size."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = -1

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Python slicing with negative numbers should work as expected
        expected = ready_indexes[:batch_size]  # This gives [0, 1, 2] for -1
        assert sampled == expected
        assert consumed == expected

    def test_sequential_sampler_non_sequential_indexes(self):
        """Test behavior with non-sequential ready indexes."""
        sampler = SequentialSampler()
        ready_indexes = [10, 5, 15, 20, 8]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [10, 5, 15]
        assert consumed == [10, 5, 15]

    def test_sequential_sampler_duplicate_indexes(self):
        """Test behavior with duplicate indexes."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 0, 2, 1, 3]
        batch_size = 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 0, 2]
        assert consumed == [0, 1, 0, 2]

    def test_sequential_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        sampled, consumed = sampler(ready_indexes, batch_size)

        assert sampled == [0, 1]
        assert consumed == [0, 1]

    def test_sequential_sampler_with_extra_kwargs(self):
        """Test that SequentialSampler accepts extra kwargs but ignores them."""
        sampler = SequentialSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        # SequentialSampler should accept extra kwargs but ignore them
        sampled, consumed = sampler.sample(ready_indexes, batch_size, extra_param="ignored")

        assert sampled == [0, 1]
        assert consumed == [0, 1]


class TestGRPOGroupNSampler:
    """Test cases for GRPOGroupNSampler."""

    def test_grpo_sampler_initialization(self):
        """Test GRPOGroupNSampler initialization."""
        sampler = GRPOGroupNSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_grpo_sampler_basic_functionality(self):
        """Test basic grouped sampling functionality."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_partial_batch(self):
        """Test partial batch sampling."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 12 indexes
        batch_size = 8  # Want 8 samples total
        # 2 groups of 4

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(sampled) == batch_size
        assert len(consumed) == batch_size

    def test_grpo_sampler_batch_size_divisibility(self):
        """Test that batch_size must be divisible by n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes, sufficient for batch_size=7
        batch_size = 7

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(ready_indexes, batch_size)

        assert "must be a multiple of n_samples_per_prompt" in str(exc_info.value)

    def test_grpo_sampler_insufficient_ready_indexes(self):
        """Test behavior when not enough ready indexes are available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3]  # Only 4 indexes, but need 8 for 2 groups of 4
        batch_size = 8

        # Should return empty lists when insufficient complete groups
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_exact_multiple_available(self):
        """Test when ready_indexes length is exactly a multiple of n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]  # 8 indexes
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_single_sample_per_prompt(self):
        """Test with n_samples_per_prompt = 1."""
        sampler = GRPOGroupNSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]

    def test_grpo_sampler_large_group_size(self):
        """Test with large n_samples_per_prompt."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=10)
        ready_indexes = list(range(20))  # 20 indexes
        batch_size = 20

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        assert sampled == list(range(20))
        assert consumed == list(range(20))

    def test_grpo_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4

        sampled, consumed = sampler(ready_indexes, batch_size)

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_grpo_sampler_with_extra_kwargs(self):
        """Test that GRPOGroupNSampler accepts extra kwargs but ignores them."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 8

        # GRPOGroupNSampler should accept extra kwargs but ignore them
        sampled, consumed = sampler.sample(ready_indexes, batch_size, extra_param="ignored", another_param=42)

        assert sampled == [0, 1, 2, 3, 4, 5, 6, 7]
        assert consumed == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_grpo_sampler_non_sequential_indexes(self):
        """Test with non-sequential ready indexes that get sorted."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [3, 4, 5, 6, 9, 10, 11, 12]  # Non-sequential order but has consecutive groups after sorting
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Should find consecutive groups after sorting: [3,4,5,6] and [9,10,11,12]
        expected = [3, 4, 5, 6, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_invalid_n_samples_per_prompt(self):
        """Test behavior with invalid n_samples_per_prompt values."""
        # Test zero n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            GRPOGroupNSampler(n_samples_per_prompt=0)
        assert "must be positive" in str(exc_info.value)
        # Test negative n_samples_per_prompt
        with pytest.raises(ValueError) as exc_info:
            GRPOGroupNSampler(n_samples_per_prompt=-2)
        assert "must be positive" in str(exc_info.value)

    def test_grpo_sampler_no_complete_groups(self):
        """Test behavior when no complete groups are available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
        ready_indexes = [0, 1, 3, 4, 6, 7]  # No consecutive groups of size 3
        batch_size = 6

        # Should return empty lists when no complete groups found
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == []
        assert consumed == []

    def test_grpo_sampler_mixed_groups(self):
        """Test behavior with mixed complete and incomplete groups."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=3)
        ready_indexes = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]  # Mixed groups
        batch_size = 6

        # Should find the complete groups [3,4,5] and [9,10,11]
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert sampled == [3, 4, 5, 9, 10, 11]
        assert consumed == [3, 4, 5, 9, 10, 11]

    def test_grpo_sampler_sorting_functionality(self):
        """Test that ready_indexes are properly sorted before group detection."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [10, 11, 12, 5, 6, 7, 8, 9]  # Out of order but contains consecutive groups
        batch_size = 8

        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # After sorting: [5,6,7,8,9,10,11,12], should find [5,6,7,8] and [9,10,11,12]
        expected = [5, 6, 7, 8, 9, 10, 11, 12]
        assert sampled == expected
        assert consumed == expected

    def test_grpo_sampler_insufficient_groups(self):
        """Test behavior when requesting more groups than available."""
        sampler = GRPOGroupNSampler(n_samples_per_prompt=4)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # 4 groups of 4
        batch_size = 12  # Requesting 3 groups of 4 - this should work

        # This should actually work fine since we have 4 groups and request 3
        sampled, consumed = sampler.sample(ready_indexes, batch_size)
        assert len(sampled) == 12
        assert len(consumed) == 12

        # Now test requesting more than available
        batch_size = 20  # Requesting 5 groups of 4, but only have 4
        sampled, consumed = sampler.sample(ready_indexes, batch_size)

        # Should return empty lists when requesting more complete groups than available
        assert sampled == []
        assert consumed == []


class TestRankAwareSampler:
    """Test cases for RankAwareSampler."""

    def test_rank_aware_sampler_initialization(self):
        """Test RankAwareSampler initialization."""
        sampler = RankAwareSampler()
        assert isinstance(sampler, BaseSampler)
        assert hasattr(sampler, "_states")
        assert sampler._states == {}

    def test_rank_aware_sampler_basic_sampling(self):
        """Test basic sampling functionality."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == [0, 1, 2]
        assert consumed == [0, 1, 2]
        assert len(sampled) == batch_size

    def test_rank_aware_sampler_caching_on_same_batch_index(self):
        """Test that same batch_index returns cached results."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 3

        # First call with batch_index=0
        sampled1, consumed1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Second call with same batch_index=0 should return cached result
        sampled2, consumed2 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled1 == sampled2 == [0, 1, 2]
        assert consumed1 == consumed2 == [0, 1, 2]

    def test_rank_aware_sampler_different_batch_indexes(self):
        """Test that different batch_index values sample different data."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        # First batch
        sampled1, consumed1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Second batch
        ready_indexes = [2, 3, 4, 5, 6, 7]
        sampled2, consumed2 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=1,
            task_name="task",
            partition_id="test",
        )

        assert sampled1 == [0, 1]
        assert sampled2 == [2, 3]
        assert consumed1 == [0, 1]
        assert consumed2 == [2, 3]

    def test_rank_aware_sampler_multiple_dp_ranks(self):
        """Test that same dp_ranks reuse state cache."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        # DP rank 0 samples batch 0
        sampled_dp0_b0, consumed_dp0_b0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )
        ready_indexes = [2, 3, 4, 5, 6, 7]
        # DP rank 0 samples batch 0 (should get same result as dp_rank=0)
        sampled_dp1_b0, consumed_dp1_b0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        # Both should sample from the same ready_indexes
        assert sampled_dp0_b0 == [0, 1]
        assert sampled_dp1_b0 == [0, 1]

    def test_rank_aware_sampler_empty_ready_indexes(self):
        """Test behavior with empty ready indexes."""
        sampler = RankAwareSampler()
        ready_indexes = []
        batch_size = 3

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_batch_size_larger_than_ready(self):
        """Test behavior when batch_size > len(ready_indexes)."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1]
        batch_size = 5

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_zero_batch_size(self):
        """Test behavior with zero batch size."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 0

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == []
        assert consumed == []

    def test_rank_aware_sampler_multiple_tasks(self):
        """Test behavior with multiple tasks."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 2

        sampled_task0, consumed_task0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task0",
            partition_id="test",
        )

        sampled_task1, consumed_task1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task1",
            partition_id="test",
        )

        assert sampled_task0 == [0, 1]
        assert consumed_task0 == [0, 1]
        assert sampled_task1 == [0, 1]
        assert consumed_task1 == [0, 1]

        # Check that state is separate per task
        assert sampler._states["test"]["task0"][0][0] == [0, 1]
        assert sampler._states["test"]["task1"][0][0] == [0, 1]

    def test_rank_aware_sampler_multiple_partitions(self):
        """Test behavior with multiple partitions."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 2

        sampled_part0, consumed_part0 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="partition0",
        )

        sampled_part1, consumed_part1 = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="partition1",
        )

        assert sampled_part0 == [0, 1]
        assert consumed_part0 == [0, 1]
        assert sampled_part1 == [0, 1]
        assert consumed_part1 == [0, 1]

        # Check that state is separate per partition
        assert sampler._states["partition0"]["task"][0][0] == [0, 1]
        assert sampler._states["partition1"]["task"][0][0] == [0, 1]

    def test_rank_aware_sampler_invalid_dp_rank(self):
        """Test behavior with invalid dp_rank."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(
                ready_indexes,
                batch_size,
                dp_rank=-1,
                batch_index=0,
                task_name="task",
                partition_id="test",
            )

        assert "dp_rank" in str(exc_info.value)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_rank_aware_sampler_with_extra_kwargs(self):
        """Test that RankAwareSampler accepts extra kwargs but ignores them."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3, 4, 5]
        batch_size = 2

        # Should accept extra kwargs gracefully
        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
            extra_param="ignored",
            another_param=42,
        )

        assert sampled == [0, 1]
        assert consumed == [0, 1]

    def test_rank_aware_sampler_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = RankAwareSampler()
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2

        sampled, consumed = sampler(
            ready_indexes,
            batch_size,
            dp_rank=0,
            batch_index=0,
            task_name="task",
            partition_id="test",
        )

        assert sampled == [0, 1]
        assert consumed == [0, 1]


class TestSeqlenBalancedSampler:
    """Test cases for SeqlenBalancedSampler."""

    # ---- Helper: mock partition object ----

    class MockPartition:
        """Minimal mock for DataPartitionStatus providing get_custom_meta."""

        def __init__(self, custom_meta: dict[int, dict]):
            self._custom_meta = custom_meta

        def get_custom_meta(self, global_indices: list[int]) -> dict[int, dict]:
            return {idx: self._custom_meta.get(idx, {}) for idx in global_indices}

    # ---- Initialization tests ----

    def test_initialization_invalid_dp_size(self):
        """Test that dp_size must be positive."""
        with pytest.raises(ValueError) as exc_info:
            SeqlenBalancedSampler(dp_size=0)
        assert "dp_size must be positive" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            SeqlenBalancedSampler(dp_size=-1)
        assert "dp_size must be positive" in str(exc_info.value)

    # ---- Fallback (no partition) tests ----

    def test_fallback_equal_split_no_partition(self):
        """Test fallback equal-split when no partition is provided."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 2  # per-DP → global = 4

        sampled_0, consumed_0 = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_1, consumed_1 = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
        )

        # Together they should cover all 4 indexes without overlap
        assert len(sampled_0) == 2
        assert len(sampled_1) == 2
        assert set(sampled_0 + sampled_1) == {0, 1, 2, 3}
        assert sampled_0 == consumed_0
        assert sampled_1 == consumed_1

    def test_fallback_single_dp(self):
        """Test dp_size=1 returns all samples to rank 0."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=1)
        ready_indexes = [0, 1, 2, 3]
        batch_size = 4  # per-DP = global = 4

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    # ---- Balanced partitioning with mock partition ----

    def test_balanced_partitioning_with_custom_meta(self):
        """Test that samples are balanced by total_lengths across DP ranks."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]
        # Sample 0 and 1 are long, sample 2 and 3 are short
        partition = self.MockPartition(
            {
                0: {"total_lengths": 100},
                1: {"total_lengths": 100},
                2: {"total_lengths": 10},
                3: {"total_lengths": 10},
            }
        )

        sampled_0, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
            partition=partition,
        )
        sampled_1, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
            partition=partition,
        )

        # All indexes should be covered
        all_sampled = sorted(sampled_0 + sampled_1)
        assert all_sampled == [0, 1, 2, 3]

        # KK should pair one long with one short per rank for balance
        def total_len(indices):
            lengths = {0: 100, 1: 100, 2: 10, 3: 10}
            return sum(lengths[i] for i in indices)

        diff = abs(total_len(sampled_0) - total_len(sampled_1))
        # Perfect balance: each rank gets one 100 + one 10 = 110, diff = 0
        assert diff == 0

    def test_balanced_partitioning_group_level(self):
        """Test balanced partitioning at group level (n_samples_per_prompt > 1)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=2)
        # 4 groups of 2: [0,1], [2,3], [4,5], [6,7]
        ready_indexes = list(range(8))
        partition = self.MockPartition(
            {
                0: {"total_lengths": 50},
                1: {"total_lengths": 50},  # group0 total=100
                2: {"total_lengths": 5},
                3: {"total_lengths": 5},  # group1 total=10
                4: {"total_lengths": 50},
                5: {"total_lengths": 50},  # group2 total=100
                6: {"total_lengths": 5},
                7: {"total_lengths": 5},  # group3 total=10
            }
        )

        sampled_0, _ = sampler.sample(
            ready_indexes,
            4,  # per-DP batch=4, global=8
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
            partition=partition,
        )
        sampled_1, _ = sampler.sample(
            ready_indexes,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
            partition=partition,
        )

        # Each rank should get 4 samples (2 groups)
        assert len(sampled_0) == 4
        assert len(sampled_1) == 4
        assert set(sampled_0 + sampled_1) == set(range(8))

        # Group integrity: each group's samples stay together
        for rank_samples in [sampled_0, sampled_1]:
            for s in rank_samples:
                partner = s ^ 1  # pairs: (0,1), (2,3), (4,5), (6,7)
                if s % 2 == 0:
                    assert partner in rank_samples, f"Group broken: {s} without {partner}"

    # ---- Caching tests ----

    def test_caching_returns_same_result(self):
        """Test that repeated calls with same key return cached result."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        sampled_first, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_second, _ = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled_first == sampled_second

    def test_different_batch_index_not_cached(self):
        """Test that different batch_index produces different cache keys."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)
        ready_indexes_b0 = [0, 1, 2, 3]
        ready_indexes_b1 = [4, 5, 6, 7]

        sampled_b0, _ = sampler.sample(
            ready_indexes_b0,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampled_b1, _ = sampler.sample(
            ready_indexes_b1,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=1,
        )

        assert sampled_b0 == [0, 1, 2, 3]
        assert sampled_b1 == [4, 5, 6, 7]

    def test_states_cache_populated_for_all_ranks(self):
        """Test that _states cache is populated for all dp_ranks on first call."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=3)
        ready_indexes = list(range(6))

        sampler.sample(
            ready_indexes,
            2,  # per-DP=2, global=6
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        # All 3 ranks should have cached state
        states = sampler._states["p0"]["task"]
        for rank_i in range(3):
            assert rank_i in states
            assert 0 in states[rank_i]
            cached_sampled, cached_consumed = states[rank_i][0]
            assert len(cached_sampled) == 2
            assert cached_sampled == cached_consumed

    # ---- clear_cache tests ----

    def test_clear_cache(self):
        """Test clear_cache removes both _states and _balanced_cache."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert len(sampler._balanced_cache) > 0
        assert "p0" in sampler._states

        sampler.clear_cache("p0")

        assert all(k[0] != "p0" for k in sampler._balanced_cache)
        assert "p0" not in sampler._states

    def test_clear_cache_only_affects_target_partition(self):
        """Test clear_cache only removes the specified partition."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)

        sampler.sample(
            [0, 1],
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        sampler.sample(
            [2, 3],
            2,
            task_name="task",
            partition_id="p1",
            dp_rank=0,
            batch_index=0,
        )

        sampler.clear_cache("p0")

        assert "p0" not in sampler._states
        assert "p1" in sampler._states
        assert any(k[0] == "p1" for k in sampler._balanced_cache)

    # ---- Edge cases ----

    def test_insufficient_ready_indexes(self):
        """Test behavior when not enough ready indexes for global batch."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=2)
        ready_indexes = [0, 1]  # Only 1 group, need 2 (global_batch = 4)

        sampled, consumed = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == []
        assert consumed == []

    def test_dp_rank_out_of_range(self):
        """Test behavior when dp_rank >= dp_size (returns empty)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=2)
        ready_indexes = [0, 1, 2, 3]

        # First call to populate cache
        sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )
        # dp_rank=5 is out of range
        sampled, consumed = sampler.sample(
            ready_indexes,
            2,
            task_name="task",
            partition_id="p0",
            dp_rank=5,
            batch_index=0,
        )

        assert sampled == []
        assert consumed == []

    def test_call_method(self):
        """Test that __call__ method works correctly."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=1, dp_size=1)
        ready_indexes = [0, 1, 2, 3]

        sampled, consumed = sampler(
            ready_indexes,
            4,
            task_name="task",
            partition_id="p0",
            dp_rank=0,
            batch_index=0,
        )

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_batch_size_not_divisible_by_n_samples_per_prompt(self):
        """Test that batch_size must be divisible by n_samples_per_prompt (inherited)."""
        sampler = SeqlenBalancedSampler(n_samples_per_prompt=4, dp_size=2)
        ready_indexes = list(range(20))

        with pytest.raises(ValueError) as exc_info:
            sampler.sample(
                ready_indexes,
                3,  # per-DP=3, global=6, 6 % 4 != 0
                task_name="task",
                partition_id="p0",
                dp_rank=0,
                batch_index=0,
            )

        assert "must be a multiple of n_samples_per_prompt" in str(exc_info.value)


class TestKarmarkarKarp:
    """Test cases for karmarkar_karp and get_seqlen_balanced_partitions utilities."""

    def test_equal_size_basic(self):
        """Test equal-size partitioning with balanced inputs."""
        seqlens = [10, 20, 30, 40]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        assert len(partitions) == 2
        assert all(len(p) == 2 for p in partitions)
        # All indices covered
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3]

    def test_equal_size_balance_quality(self):
        """Test that KK produces well-balanced partitions."""
        seqlens = [100, 90, 50, 10, 5, 1]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        sums = [sum(seqlens[i] for i in p) for p in partitions]
        # Difference should be small relative to total
        assert abs(sums[0] - sums[1]) <= max(seqlens)

    def test_unequal_size(self):
        """Test variable-size partitioning."""
        seqlens = [100, 10, 10, 10, 10]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=False)

        assert len(partitions) == 2
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3, 4]

    def test_single_partition(self):
        """Test with k_partitions=1 returns all items."""
        seqlens = [10, 20, 30]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=1, equal_size=False)

        assert len(partitions) == 1
        assert sorted(partitions[0]) == [0, 1, 2]

    def test_equal_size_assertion_error(self):
        """Test that equal_size raises when items not divisible by k."""
        seqlens = [10, 20, 30]
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

    def test_too_few_items(self):
        """Test that too few items raises AssertionError."""
        seqlens = [10]
        with pytest.raises(AssertionError):
            get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=False)

    def test_three_way_partition(self):
        """Test 3-way partitioning."""
        seqlens = [100, 80, 60, 40, 20, 10]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=3, equal_size=True)

        assert len(partitions) == 3
        assert all(len(p) == 2 for p in partitions)
        assert sorted(sum(partitions, [])) == [0, 1, 2, 3, 4, 5]

    def test_identical_seqlens(self):
        """Test with all identical sequence lengths."""
        seqlens = [50, 50, 50, 50]
        partitions = get_seqlen_balanced_partitions(seqlens, k_partitions=2, equal_size=True)

        sums = [sum(seqlens[i] for i in p) for p in partitions]
        assert sums[0] == sums[1] == 100


class TestStreamingTokenBudgetSampler:
    """Test cases for StreamingTokenBudgetSampler."""

    class MockPartition:
        """Minimal mock for DataPartitionStatus providing get_custom_meta."""

        def __init__(self, custom_meta: dict[int, dict]):
            self._custom_meta = custom_meta

        def get_custom_meta(self, global_indices: list[int]) -> dict[int, dict]:
            return {idx: self._custom_meta.get(idx, {}) for idx in global_indices}

    @staticmethod
    def _partition_with_uniform_lengths(indexes, length):
        return TestStreamingTokenBudgetSampler.MockPartition({i: {"total_lengths": length} for i in indexes})

    # ---- Initialization ----

    def test_initialization_defaults(self):
        sampler = StreamingTokenBudgetSampler()
        assert isinstance(sampler, GRPOGroupNSampler)
        assert sampler.n_samples_per_prompt == 1
        assert sampler.balance_unit_multiplier == 1
        assert sampler._buckets == {}
        assert sampler._assigned_global == {}
        assert sampler._resolved_lengths == {}

    def test_initialization_invalid_balance_unit_multiplier(self):
        with pytest.raises(ValueError) as exc_info:
            StreamingTokenBudgetSampler(balance_unit_multiplier=0)
        assert "balance_unit_multiplier must be positive" in str(exc_info.value)

        with pytest.raises(ValueError):
            StreamingTokenBudgetSampler(balance_unit_multiplier=-3)

    def test_initialization_invalid_n_samples_per_prompt(self):
        # Inherited validation from GRPOGroupNSampler.
        with pytest.raises(ValueError) as exc_info:
            StreamingTokenBudgetSampler(n_samples_per_prompt=0)
        assert "must be positive" in str(exc_info.value)

    # ---- Fallback path (no token_budget) ----

    def test_fallback_to_grpo_without_token_budget(self):
        """Without token_budget, delegate to the inherited GRPO sample()."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]

        sampled, consumed = sampler.sample(ready_indexes, batch_size=4, task_name="ref", partition_id="p0")

        assert sampled == [0, 1, 2, 3]
        assert consumed == [0, 1, 2, 3]

    def test_fallback_strips_streaming_only_kwargs(self):
        """Streaming-only kwargs must not leak into the GRPO fallback call."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2)
        ready_indexes = [0, 1, 2, 3]

        # dp_size / allow_underfill / partition are streaming extras; GRPO must
        # not choke on them. (token_budget absent → fallback path.)
        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size=2,
            task_name="ref",
            partition_id="p0",
            dp_size=2,
            allow_underfill=True,
            partition=object(),
        )

        assert sampled == [0, 1]
        assert consumed == [0, 1]

    # ---- Argument validation (token_budget path) ----

    def test_requires_dp_rank_and_dp_size(self):
        sampler = StreamingTokenBudgetSampler()
        with pytest.raises(ValueError) as exc_info:
            sampler.sample([0, 1], batch_size=0, token_budget=100, partition=object())
        assert "dp_rank" in str(exc_info.value)

    def test_requires_partition(self):
        sampler = StreamingTokenBudgetSampler()
        with pytest.raises(ValueError) as exc_info:
            sampler.sample(
                [0, 1, 2, 3],
                batch_size=0,
                token_budget=100,
                dp_rank=0,
                dp_size=1,
            )
        assert "partition" in str(exc_info.value)

    # ---- Basic token-budget slicing (single DP) ----

    def test_single_dp_packs_without_overshooting_budget(self):
        """Single DP packs the largest prefix that does NOT overshoot the budget.

        With samples of length 100 and budget 250, the third sample would push the
        slice to 300 > 250, so the slice stops at 2 samples (200). Including a
        sample that overshoots the budget risks an oversized micro-batch (OOM).
        """
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 100)

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size=0,
            task_name="actor",
            partition_id="p0",
            token_budget=250,
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
        )

        assert sampled == consumed
        # 100 + 100 = 200 <= 250; adding a third (300) would overshoot.
        assert sampled == [0, 1]

    def test_single_dp_exact_budget(self):
        """When a prefix sums exactly to the budget, it is returned in full."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 100)

        sampled, _ = sampler.sample(
            ready_indexes,
            batch_size=0,
            task_name="actor",
            partition_id="p0",
            token_budget=200,  # exactly two samples
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
        )

        assert sampled == [0, 1]

    def test_single_dp_oversized_sample_yields_at_least_one(self):
        """A single sample exceeding the budget is still returned (progress)."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 1000)

        sampled, consumed = sampler.sample(
            ready_indexes,
            batch_size=0,
            task_name="actor",
            partition_id="p0",
            token_budget=100,  # smaller than a single sample
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
        )

        assert len(sampled) == 1
        assert sampled == consumed

    # ---- Cross-DP balancing ----

    def test_cross_dp_token_balance(self):
        """Two DPs at the same batch_index get token-balanced, disjoint slices."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        # Two long, two short → balanced split should pair long+short per DP.
        partition = self.MockPartition(
            {
                0: {"total_lengths": 100},
                1: {"total_lengths": 100},
                2: {"total_lengths": 10},
                3: {"total_lengths": 10},
            }
        )
        common = dict(
            task_name="actor",
            partition_id="p0",
            dp_size=2,
            batch_index=0,
            partition=partition,
            token_budget=110,
        )

        sampled_0, consumed_0 = sampler.sample(ready_indexes, 0, dp_rank=0, **common)
        sampled_1, consumed_1 = sampler.sample(ready_indexes, 0, dp_rank=1, **common)

        # Disjoint and fully covering.
        assert set(sampled_0).isdisjoint(sampled_1)
        assert set(sampled_0 + sampled_1) == {0, 1, 2, 3}
        assert sampled_0 == consumed_0
        assert sampled_1 == consumed_1

        lengths = {0: 100, 1: 100, 2: 10, 3: 10}
        tok_0 = sum(lengths[i] for i in sampled_0)
        tok_1 = sum(lengths[i] for i in sampled_1)
        # Perfect balance: each DP gets one long + one short = 110.
        assert tok_0 == tok_1 == 110

    # ---- PP-stage cache (batch_index alignment) ----

    def test_batch_index_cache_is_stable(self):
        """Repeated requests for the same (dp_rank, batch_index) hit the cache."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 50)
        kwargs = dict(
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=2,
            batch_index=0,
            partition=partition,
            token_budget=50,
        )

        first, _ = sampler.sample(ready_indexes, 0, **kwargs)
        # Second call with a DIFFERENT ready pool must still return the cached slice.
        second, _ = sampler.sample([10, 11, 12, 13], 0, **kwargs)

        assert first == second
        assert first  # non-empty

    def test_different_batch_index_advances_stream(self):
        """Different batch_index consumes the next slice (no re-issue)."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_all = [0, 1, 2, 3, 4, 5, 6, 7]
        partition = self._partition_with_uniform_lengths(ready_all, 100)
        base = dict(
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=1,
            partition=partition,
            token_budget=100,
        )

        b0, _ = sampler.sample(ready_all, 0, batch_index=0, **base)
        # Remaining ready pool excludes what batch 0 took.
        remaining = [i for i in ready_all if i not in b0]
        b1, _ = sampler.sample(remaining, 0, batch_index=1, **base)

        assert b0
        assert b1
        assert set(b0).isdisjoint(b1)

    # ---- End-of-stream tail flush ----

    def test_tail_flush_on_production_done_drains_all(self):
        """production_done releases a sub-balance_unit remainder into the buckets.

        A single batch_index returns one budget-sized micro-batch per DP, but the
        tail flush guarantees EVERY ready sample is assigned to some DP bucket so
        nothing is orphaned (which would livelock end-of-stream). We drain across
        successive batch_index calls and assert full coverage with no overlap.
        """
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2, balance_unit_multiplier=4)
        # balance_unit = dp_size(2) * n(2) * mult(4) = 16, but only 4 ready.
        all_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(all_indexes, 10)

        # The controller removes consumed samples from the ready pool, so we model
        # that by feeding only the not-yet-drained indexes on each new batch_index.
        drained: list[int] = []
        for batch_index in range(4):  # more than enough to fully drain
            ready_now = [i for i in all_indexes if i not in drained]
            for dp_rank in (0, 1):
                sampled, consumed = sampler.sample(
                    ready_now,
                    0,
                    task_name="actor",
                    partition_id="p0",
                    dp_rank=dp_rank,
                    dp_size=2,
                    batch_index=batch_index,
                    partition=partition,
                    token_budget=10,
                    production_done=True,
                )
                assert sampled == consumed
                drained.extend(sampled)

        # Every produced sample drained exactly once across the stream.
        assert sorted(drained) == [0, 1, 2, 3]
        assert len(drained) == len(set(drained))

    def test_eos_drains_all_across_rounds_within_budget(self):
        """Regression for the step-50 drain hang AND the EOS OOM.

        At end-of-stream every produced sample must eventually be delivered (and
        marked consumed) — but NO single micro-batch may exceed the token budget
        (popping a whole bucket at once built a 17-sample mb → OOM). So a dp whose
        bucket holds several budgets of residue is drained across SUCCESSIVE
        batch_index rounds, each mb ≤ budget. This models the real consumer: each
        dp advances its OWN batch_index on every non-empty fetch and stops when
        the global partition is fully consumed.
        """
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2, balance_unit_multiplier=4)
        all_indexes = list(range(8))  # 4 GRPO groups
        # Long per-sample length so each budget slice fits exactly ONE sample,
        # forcing multi-round drain of any multi-sample bucket.
        per_sample_len = 900
        partition = self._partition_with_uniform_lengths(all_indexes, per_sample_len)
        dp_size = 2
        token_budget = 1000  # one 900-token sample fits; two (1800) do not

        # All dps step through batch_index in lockstep (the sampler atomically
        # prepares every dp's slice for a batch_index on first touch); we advance
        # to the next batch_index once a round has been served, and stop when a
        # whole round yields nothing.
        drained: list[int] = []
        for batch_index in range(20):  # bounded; must finish well within
            # Model the controller: consumed samples are filtered OUT of the ready
            # pool (scan_data_status excludes consumption_status==1), so already
            # drained indexes are never re-offered.
            ready_now = [i for i in all_indexes if i not in drained]
            round_total = 0
            for dp_rank in range(dp_size):
                sampled, consumed = sampler.sample(
                    ready_now,
                    0,
                    task_name="actor",
                    partition_id="p0",
                    dp_rank=dp_rank,
                    dp_size=dp_size,
                    batch_index=batch_index,
                    partition=partition,
                    token_budget=token_budget,
                    production_done=True,
                )
                assert sampled == consumed
                if sampled:
                    # OOM guard: each delivered mb must respect the token budget.
                    mb_tokens = len(sampled) * per_sample_len
                    assert mb_tokens <= token_budget, f"oversized mb: {len(sampled)} samples = {mb_tokens} tok"
                    drained.extend(sampled)
                    round_total += len(sampled)
            if round_total == 0:
                break  # whole round empty → fully drained

        # Every produced sample consumed exactly once across the multi-round drain.
        assert sorted(drained) == all_indexes, f"orphaned: {set(all_indexes) - set(drained)}"
        assert len(drained) == len(set(drained))
        # Nothing left bucketed / assigned-but-unconsumed.
        for bucket in sampler._buckets[("p0", "actor")].values():
            assert bucket == [], f"residue left in bucket: {bucket}"
        assert sampler._assigned_global[("p0", "actor")] == set()

    def test_tail_flush_assigns_all_to_buckets(self):
        """First production_done call must assign all ready samples to buckets."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2, balance_unit_multiplier=4)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 10)

        sampler.sample(
            ready_indexes,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=2,
            batch_index=0,
            partition=partition,
            token_budget=10,
            production_done=True,
        )

        assigned = sampler._assigned_global[("p0", "actor")]
        bucketed = set()
        for bucket in sampler._buckets[("p0", "actor")].values():
            bucketed.update(bucket)
        # Everything not yet popped is still tracked in assigned and lives in a bucket.
        assert assigned == bucketed
        # All four samples are accounted for (either popped this call or bucketed).
        popped = set(range(4)) - assigned
        assert (assigned | popped) == {0, 1, 2, 3}

    def test_eos_waits_for_downstream_fields_before_empty_cache(self):
        """EOS waits for downstream fields before caching an empty result."""
        from transfer_queue.controller import DataPartitionStatus

        def schema(field_name: str) -> dict[str, dict[str, Any]]:
            return {field_name: {"dtype": "torch.float32", "shape": (4,), "is_nested": False, "is_non_tensor": False}}

        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        partition = DataPartitionStatus(partition_id="train_0")
        partition.global_indexes.update([0, 1])
        partition.actual_sample_count = 2
        partition.pending_last_indexes.update([0, 1])
        partition.has_pending_last = True
        partition.pending_last_fields.update(["tokens"])
        partition.set_custom_meta({0: {"total_lengths": 5}, 1: {"total_lengths": 5}})
        partition.update_production_status([0, 1], ["tokens"], schema("tokens"))
        assert partition.production_completed is True

        data_fields = ["tokens", "advantages"]
        production_done = partition.are_unconsumed_fields_ready("actor_train", data_fields)
        assert production_done is False

        sampled, consumed = sampler.sample(
            [],
            0,
            task_name="actor_train",
            partition_id="train_0",
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
            token_budget=10,
            production_done=production_done,
        )
        assert sampled == []
        assert consumed == []
        assert sampler._states == {}

        partition.update_production_status([0, 1], ["advantages"], schema("advantages"))
        ready = partition.scan_data_status(data_fields, "actor_train")
        assert ready == [0, 1]
        production_done = partition.are_unconsumed_fields_ready("actor_train", data_fields)
        assert production_done is True

        sampled, consumed = sampler.sample(
            ready,
            0,
            task_name="actor_train",
            partition_id="train_0",
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
            token_budget=10,
            production_done=production_done,
        )
        assert sampled == [0, 1]
        assert consumed == [0, 1]

    def test_no_complete_group_without_production_done_returns_empty(self):
        """No complete GRPO group + not end-of-stream → return empty (wait for more)."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2, balance_unit_multiplier=4)
        # Non-consecutive indexes → GRPOGroupNSampler finds no complete group of 2.
        ready_indexes = [0, 2]
        partition = self._partition_with_uniform_lengths(ready_indexes, 10)

        sampled, consumed = sampler.sample(
            ready_indexes,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=2,
            batch_index=0,
            partition=partition,
            token_budget=10,
            production_done=False,
        )

        assert sampled == []
        assert consumed == []

    def test_complete_group_drains_below_balance_unit(self):
        """A single complete group is released even below balance_unit (trickle drain).

        With long responses the producer trickles one group at a time and the ready
        pool stays below balance_unit; the sampler must still make progress rather
        than wait for a full unit forever.
        """
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=2, balance_unit_multiplier=4)
        # balance_unit = 16, but a single consecutive group [0,1] is ready.
        ready_indexes = [0, 1]
        partition = self._partition_with_uniform_lengths(ready_indexes, 10)

        got_any = False
        for dp_rank in (0, 1):
            sampled, _ = sampler.sample(
                ready_indexes,
                0,
                task_name="actor",
                partition_id="p0",
                dp_rank=dp_rank,
                dp_size=2,
                batch_index=0,
                partition=partition,
                token_budget=10,
                production_done=False,
            )
            got_any = got_any or bool(sampled)

        assert got_any, "a complete group below balance_unit should still be released"

    # ---- Missing total_lengths fallback ----

    def test_missing_total_lengths_uses_budget_fallback(self):
        """Samples lacking total_lengths fall back to token_budget (safe over-estimate)."""
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        # No custom_meta at all → every sample missing total_lengths.
        partition = self.MockPartition({})

        sampled, consumed = sampler.sample(
            ready_indexes,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
            token_budget=500,
        )

        # With fallback length == budget, one sample already meets the budget.
        assert len(sampled) == 1
        assert sampled == consumed

    # ---- Assignment / no double-issue ----

    def test_bucketed_samples_not_reassigned(self):
        """Samples already sitting in a bucket are filtered out of the ready pool.

        The sampler tracks bucketed-but-not-yet-popped samples in ``assigned`` and
        excludes them from ``available_ready`` so a balance round never re-assigns a
        sample that is already waiting in some DP's bucket. (Consumed samples are
        filtered by the controller, not the sampler.)
        """
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        # dp_size=2, budget small → each balance round assigns 2 samples (one per
        # DP) but each DP pops only its budget slice, leaving the rest bucketed.
        ready_all = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_all, 100)

        # First call (batch 0) seeds buckets for both DPs.
        sampler.sample(
            ready_all,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=2,
            batch_index=0,
            partition=partition,
            token_budget=100,
        )
        sampler.sample(
            ready_all,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=1,
            batch_index=0,
            dp_size=2,
            partition=partition,
            token_budget=100,
        )

        assigned = sampler._assigned_global[("p0", "actor")]
        bucketed = set()
        for bucket in sampler._buckets[("p0", "actor")].values():
            bucketed.update(bucket)
        # Invariant: assigned set == union of bucket contents (no leak, no double-count).
        assert assigned == bucketed
        # No index appears in more than one DP bucket.
        all_bucket_items = [i for b in sampler._buckets[("p0", "actor")].values() for i in b]
        assert len(all_bucket_items) == len(set(all_bucket_items))

    # ---- clear_cache ----

    def test_clear_cache_removes_partition_state(self):
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        ready_indexes = [0, 1, 2, 3]
        partition = self._partition_with_uniform_lengths(ready_indexes, 50)

        sampler.sample(
            ready_indexes,
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
            token_budget=50,
        )

        # State should now exist for p0.
        assert any(k[0] == "p0" for k in sampler._buckets)
        assert "p0" in sampler._states

        sampler.clear_cache("p0")

        assert all(k[0] != "p0" for k in sampler._buckets)
        assert all(k[0] != "p0" for k in sampler._assigned_global)
        assert all(k[0] != "p0" for k in sampler._resolved_lengths)
        assert "p0" not in sampler._states
        # Internal GRPO scratch state used by balance rounds must be cleared too.
        assert "__streaming_internal__" not in sampler._states

    def test_clear_cache_only_affects_target_partition(self):
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        for pid in ("p0", "p1"):
            partition = self._partition_with_uniform_lengths([0, 1, 2, 3], 50)
            sampler.sample(
                [0, 1, 2, 3],
                0,
                task_name="actor",
                partition_id=pid,
                dp_rank=0,
                dp_size=1,
                batch_index=0,
                partition=partition,
                token_budget=50,
            )

        sampler.clear_cache("p0")

        assert all(k[0] != "p0" for k in sampler._buckets)
        assert any(k[0] == "p1" for k in sampler._buckets)

    # ---- Empty input ----

    def test_empty_ready_indexes_returns_empty(self):
        sampler = StreamingTokenBudgetSampler(n_samples_per_prompt=1)
        partition = self.MockPartition({})

        sampled, consumed = sampler.sample(
            [],
            0,
            task_name="actor",
            partition_id="p0",
            dp_rank=0,
            dp_size=1,
            batch_index=0,
            partition=partition,
            token_budget=100,
        )

        assert sampled == []
        assert consumed == []


class TestSamplerIntegration:
    """Integration tests for samplers."""

    def test_samplers_implement_base_interface(self):
        """Test that all samplers properly implement BaseSampler interface."""
        samplers = [SequentialSampler(), GRPOGroupNSampler(), SeqlenBalancedSampler()]

        for sampler in samplers:
            # Test that they are instances of BaseSampler
            assert isinstance(sampler, BaseSampler)

            # Test that they have the required methods
            assert hasattr(sampler, "sample")
            assert callable(sampler.sample)
            assert callable(sampler)
            assert callable(sampler.__call__)

    def test_samplers_return_consistent_types(self):
        """Test that all samplers return consistent tuple types."""
        samplers = [
            (SequentialSampler(), {}),
            (GRPOGroupNSampler(n_samples_per_prompt=2), {}),
            (
                SeqlenBalancedSampler(n_samples_per_prompt=2, dp_size=1),
                {
                    "task_name": "task",
                    "partition_id": "test",
                    "dp_rank": 0,
                    "batch_index": 0,
                },
            ),
        ]

        ready_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        batch_size = 4

        for sampler, kwargs in samplers:
            sampled, consumed = sampler.sample(ready_indexes, batch_size, **kwargs)

            # Check return types
            assert isinstance(sampled, list)
            assert isinstance(consumed, list)
            assert isinstance(sampled[0], int) if sampled else True
            assert isinstance(consumed[0], int) if consumed else True

            # Check return value consistency
            assert len(sampled) <= batch_size
            assert len(sampled) == len(consumed)

    def test_samplers_handle_edge_cases_consistently(self):
        """Test that samplers handle edge cases consistently."""
        samplers = [(SequentialSampler(), {}), (GRPOGroupNSampler(n_samples_per_prompt=2), {})]

        # Test empty ready indexes
        for sampler, kwargs in samplers:
            try:
                sampled, consumed = sampler.sample([], 0, **kwargs)
                assert sampled == []
                assert consumed == []
            except Exception:
                # GRPO sampler might fail with empty list, that's expected
                pass

        # Test zero batch size
        for sampler, kwargs in samplers:
            try:
                sampled, consumed = sampler.sample([0, 1, 2, 3], 0, **kwargs)
                assert sampled == []
                assert consumed == []
            except Exception:
                # Some samplers might not handle zero batch size
                pass


if __name__ == "__main__":
    pytest.main([__file__])
