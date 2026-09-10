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

import math
from contextlib import ExitStack
from typing import Any

import ray
from omegaconf import DictConfig

from transfer_queue.storage.bootstrap.provider import StorageBootstrapProvider
from transfer_queue.storage.simple_storage import SimpleStorageUnit
from transfer_queue.utils.common import get_node_round_robin_scheduling_strategies, get_placement_group
from transfer_queue.utils.logging_utils import get_logger
from transfer_queue.utils.zmq_utils import process_zmq_server_info

logger = get_logger(__name__)


@StorageBootstrapProvider.register_provider("SimpleStorage")
def initialize_simple_storage(
    conf: DictConfig, reserved_cpus: dict[str, float] | None = None, rollback: ExitStack | None = None
) -> dict[str, Any]:
    """Initialize storage, accounting for CPU allocations and tracking owned resources for rollback."""

    simple_storage_handles = {}
    num_data_storage_units = conf.backend.SimpleStorage.num_data_storage_units
    total_storage_size = conf.backend.SimpleStorage.get("total_storage_size", None)
    required_node_resource = conf.backend.SimpleStorage.get("required_node_resource", None)
    if required_node_resource is None:
        storage_placement_group = get_placement_group(num_data_storage_units, num_cpus_per_actor=1)
        if rollback is not None:
            rollback.callback(ray.util.remove_placement_group, storage_placement_group)
        scheduling_strategies = None
    else:
        storage_placement_group = None
        scheduling_strategies = get_node_round_robin_scheduling_strategies(
            num_data_storage_units,
            required_node_resource=required_node_resource,
            reserved_cpus=reserved_cpus,
        )

    # Compute per-unit capacity: None means unlimited
    storage_unit_size = (
        math.ceil(total_storage_size / num_data_storage_units) if total_storage_size is not None else None
    )

    for storage_unit_rank in range(num_data_storage_units):
        actor_options: dict[str, Any] = {"name": f"TransferQueueStorageUnit#{storage_unit_rank}"}
        if scheduling_strategies is None:
            actor_options.update(
                placement_group=storage_placement_group,
                placement_group_bundle_index=storage_unit_rank,
            )
        else:
            strategy = scheduling_strategies[storage_unit_rank]
            actor_options["scheduling_strategy"] = strategy
            logger.info(
                f"Applying node affinity: actor={actor_options['name']} "
                f"required_node_resource={required_node_resource} node_id={strategy.node_id} "
                f"soft={str(strategy.soft).lower()}"
            )

        storage_node = SimpleStorageUnit.options(**actor_options).remote(  # type: ignore[attr-defined]
            storage_unit_size=storage_unit_size,
        )
        if rollback is not None:
            rollback.callback(ray.kill, storage_node, no_restart=True)
        simple_storage_handles[f"TransferQueueStorageUnit#{storage_unit_rank}"] = storage_node
        logger.info(f"TransferQueueStorageUnit#{storage_unit_rank} has been created.")

    storage_zmq_info = process_zmq_server_info(simple_storage_handles)
    backend_name = conf.backend.storage_backend
    conf.backend[backend_name].zmq_info = storage_zmq_info

    return simple_storage_handles
