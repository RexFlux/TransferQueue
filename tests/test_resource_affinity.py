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

from unittest.mock import MagicMock, call

import pytest
from omegaconf import OmegaConf

from transfer_queue import interface
from transfer_queue.storage.bootstrap import simple_storage_bootstrap
from transfer_queue.utils import common

_NODE_A = "01" * 28
_NODE_B = "02" * 28
_NODE_C = "03" * 28
_NODE_D = "04" * 28
_UNSET = object()


def _node(node_id: str, *, alive: bool = True, resources: dict[str, float] | None = None) -> dict:
    return {"NodeID": node_id, "Alive": alive, "Resources": resources or {}}


def _simple_storage_conf(required_node_resource=_UNSET):
    simple_storage = {
        "num_data_storage_units": 2,
        "total_storage_size": None,
    }
    if required_node_resource is not _UNSET:
        simple_storage["required_node_resource"] = required_node_resource
    return OmegaConf.create(
        {
            "backend": {
                "storage_backend": "SimpleStorage",
                "SimpleStorage": simple_storage,
            }
        }
    )


def _mock_storage_initialization(monkeypatch):
    storage_unit = MagicMock()
    storage_unit.options.return_value.remote.side_effect = [MagicMock(), MagicMock()]
    monkeypatch.setattr(simple_storage_bootstrap, "SimpleStorageUnit", storage_unit)
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", lambda _: {})
    return storage_unit


def _mock_controller_initialization(monkeypatch):
    controller = MagicMock()
    controller_handle = MagicMock()
    controller.options.return_value.remote.return_value = controller_handle
    monkeypatch.setattr(interface, "TransferQueueController", controller)
    monkeypatch.setattr(interface, "_init_from_existing", lambda: False)
    monkeypatch.setattr(interface, "_maybe_create_tq_storage", lambda conf: conf)
    monkeypatch.setattr(interface, "_maybe_create_tq_client", MagicMock())
    monkeypatch.setattr(interface, "process_zmq_server_info", lambda _: {})
    monkeypatch.setattr(interface.ray, "get", lambda value: value)
    interface._TQ_CONTROLLER = None
    interface._TQ_STORAGE = None
    interface._TQ_CLIENT = None
    return controller


@pytest.fixture(autouse=True)
def _reset_interface_globals():
    yield
    interface._TQ_CONTROLLER = None
    interface._TQ_STORAGE = None
    interface._TQ_CLIENT = None


def test_affinity_filters_dead_zero_and_missing_resources(monkeypatch):
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [
            _node(_NODE_C, resources={"storage_pool": 2}),
            _node(_NODE_B, resources={"storage_pool": 0}),
            _node(_NODE_A, resources={"storage_pool": 1}),
            _node(_NODE_D, alive=False, resources={"storage_pool": 1}),
            _node("05" * 28, resources={"compute_pool": 1}),
        ],
    )

    strategies = common.get_node_round_robin_scheduling_strategies(5, required_node_resource="storage_pool")

    assert [strategy.node_id for strategy in strategies] == [
        _NODE_A,
        _NODE_C,
        _NODE_A,
        _NODE_C,
        _NODE_A,
    ]
    assert all(strategy.soft is False for strategy in strategies)


def test_affinity_fails_fast_when_no_alive_node_matches(monkeypatch):
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [
            _node(_NODE_A, resources={"control_pool": 0}),
            _node(_NODE_B, alive=False, resources={"control_pool": 1}),
        ],
    )

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'control_pool'"):
        common.get_node_round_robin_scheduling_strategies(1, required_node_resource="control_pool")


@pytest.mark.parametrize("required_node_resource", [_UNSET, None], ids=["missing", "null"])
def test_unconfigured_simple_storage_preserves_placement_group(monkeypatch, required_node_resource):
    storage_unit = _mock_storage_initialization(monkeypatch)
    placement_group = MagicMock()
    get_placement_group = MagicMock(return_value=placement_group)
    get_strategies = MagicMock()
    monkeypatch.setattr(simple_storage_bootstrap, "get_placement_group", get_placement_group)
    monkeypatch.setattr(
        simple_storage_bootstrap,
        "get_node_round_robin_scheduling_strategies",
        get_strategies,
    )

    simple_storage_bootstrap.initialize_simple_storage(_simple_storage_conf(required_node_resource))

    get_placement_group.assert_called_once_with(2, num_cpus_per_actor=1)
    get_strategies.assert_not_called()
    assert storage_unit.options.call_args_list == [
        call(
            name="TransferQueueStorageUnit#0",
            placement_group=placement_group,
            placement_group_bundle_index=0,
        ),
        call(
            name="TransferQueueStorageUnit#1",
            placement_group=placement_group,
            placement_group_bundle_index=1,
        ),
    ]


def test_simple_storage_uses_hard_affinity_when_configured(monkeypatch):
    storage_unit = _mock_storage_initialization(monkeypatch)
    get_placement_group = MagicMock()
    monkeypatch.setattr(simple_storage_bootstrap, "get_placement_group", get_placement_group)
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [_node(_NODE_A, resources={"storage_pool": 1})],
    )

    simple_storage_bootstrap.initialize_simple_storage(_simple_storage_conf("storage_pool"))

    get_placement_group.assert_not_called()
    strategies = [options.kwargs["scheduling_strategy"] for options in storage_unit.options.call_args_list]
    assert [strategy.node_id for strategy in strategies] == [_NODE_A, _NODE_A]
    assert all(strategy.soft is False for strategy in strategies)


def test_simple_storage_fails_before_actor_creation_when_no_node_matches(monkeypatch):
    storage_unit = _mock_storage_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'storage_pool'"):
        simple_storage_bootstrap.initialize_simple_storage(_simple_storage_conf("storage_pool"))

    storage_unit.options.assert_not_called()


@pytest.mark.parametrize("controller_conf", [None, {"required_node_resource": None}])
def test_unconfigured_controller_preserves_default_ray_scheduling(monkeypatch, controller_conf):
    controller = _mock_controller_initialization(monkeypatch)
    conf = None if controller_conf is None else OmegaConf.create({"controller": controller_conf})

    interface.init(conf)

    controller.options.assert_called_once_with(
        name="TransferQueueController",
        namespace="transfer_queue",
    )


def test_controller_uses_hard_affinity_when_configured(monkeypatch):
    controller = _mock_controller_initialization(monkeypatch)
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [_node(_NODE_A, resources={"control_pool": 1})],
    )

    interface.init(OmegaConf.create({"controller": {"required_node_resource": "control_pool"}}))

    options = controller.options.call_args.kwargs
    assert options["name"] == "TransferQueueController"
    assert options["namespace"] == "transfer_queue"
    assert options["scheduling_strategy"].node_id == _NODE_A
    assert options["scheduling_strategy"].soft is False


def test_controller_fails_before_actor_creation_when_no_node_matches(monkeypatch):
    controller = _mock_controller_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'control_pool'"):
        interface.init(OmegaConf.create({"controller": {"required_node_resource": "control_pool"}}))

    controller.options.assert_not_called()
