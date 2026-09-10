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

import logging
from unittest.mock import MagicMock, call

import pytest
from omegaconf import OmegaConf
from ray.exceptions import ActorUnschedulableError, RayActorError

from transfer_queue import interface
from transfer_queue.storage.bootstrap import simple_storage_bootstrap
from transfer_queue.utils import common

_NODE_A = "01" * 28
_NODE_B = "02" * 28
_NODE_C = "03" * 28
_NODE_D = "04" * 28
_UNSET = object()


def _node(node_id: str, *, alive: bool = True, resources: dict[str, float] | None = None) -> dict:
    return {"NodeID": node_id, "Alive": alive, "Resources": {"CPU": 8, **(resources or {})}}


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
    monkeypatch.setattr(interface, "_maybe_create_tq_storage", lambda conf, **kwargs: conf)
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
def test_unconfigured_simple_storage_preserves_placement_group(monkeypatch, caplog, required_node_resource):
    caplog.set_level(logging.INFO, logger=simple_storage_bootstrap.logger.name)
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
    assert "Applying node affinity:" not in caplog.text
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


def test_simple_storage_uses_hard_affinity_when_configured(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=simple_storage_bootstrap.logger.name)
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
    affinity_logs = [record for record in caplog.records if "Applying node affinity:" in record.getMessage()]
    assert len(affinity_logs) == 2
    assert "has been created on node" not in caplog.text
    for rank, record in enumerate(affinity_logs):
        assert record.levelno == logging.INFO
        assert record.getMessage() == (
            f"Applying node affinity: actor=TransferQueueStorageUnit#{rank} "
            f"required_node_resource=storage_pool node_id={_NODE_A} soft=false"
        )


def test_simple_storage_fails_before_actor_creation_when_no_node_matches(monkeypatch):
    storage_unit = _mock_storage_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'storage_pool'"):
        simple_storage_bootstrap.initialize_simple_storage(_simple_storage_conf("storage_pool"))

    storage_unit.options.assert_not_called()


@pytest.mark.parametrize("controller_conf", [None, {"required_node_resource": None}])
def test_unconfigured_controller_preserves_default_ray_scheduling(monkeypatch, caplog, controller_conf):
    caplog.set_level(logging.INFO, logger=interface.logger.name)
    controller = _mock_controller_initialization(monkeypatch)
    conf = None if controller_conf is None else OmegaConf.create({"controller": controller_conf})

    interface.init(conf)

    controller.options.assert_called_once_with(
        name="TransferQueueController",
        namespace="transfer_queue",
    )
    assert "Applying node affinity:" not in caplog.text


def test_controller_uses_hard_affinity_when_configured(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=interface.logger.name)
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
    affinity_logs = [record for record in caplog.records if "Applying node affinity:" in record.getMessage()]
    assert len(affinity_logs) == 1
    assert affinity_logs[0].levelno == logging.INFO
    assert affinity_logs[0].getMessage() == (
        f"Applying node affinity: actor=TransferQueueController "
        f"required_node_resource=control_pool node_id={_NODE_A} soft=false"
    )


def test_controller_fails_before_actor_creation_when_no_node_matches(monkeypatch):
    controller = _mock_controller_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [])

    with pytest.raises(ValueError, match="No alive Ray nodes provide custom resource 'control_pool'"):
        interface.init(OmegaConf.create({"controller": {"required_node_resource": "control_pool"}}))

    controller.options.assert_not_called()


def test_affinity_respects_heterogeneous_cpu_capacity(monkeypatch):
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [
            _node(_NODE_A, resources={"storage_pool": 1, "CPU": 1}),
            _node(_NODE_B, resources={"storage_pool": 1, "CPU": 3}),
        ],
    )
    strategies = common.get_node_round_robin_scheduling_strategies(4, "storage_pool")
    assert [strategy.node_id for strategy in strategies] == [_NODE_A, _NODE_B, _NODE_B, _NODE_B]


def test_affinity_rejects_insufficient_cpu_capacity(monkeypatch):
    monkeypatch.setattr(
        common.ray,
        "nodes",
        lambda: [
            _node(_NODE_A, resources={"storage_pool": 1, "CPU": 1}),
        ],
    )
    with pytest.raises(ValueError, match="CPU"):
        common.get_node_round_robin_scheduling_strategies(2, "storage_pool")


def _mock_full_initialization(monkeypatch):
    """Mock Ray transport, keeping init, attach and storage bootstrap real."""
    controller = MagicMock()
    handle = controller.options.return_value.remote.return_value
    handle.get_config.remote.return_value = None
    handle.get_node_id.remote.return_value = _NODE_A
    registry = {}

    def create_controller(**kwargs):
        registry["controller"] = handle
        return handle

    def get_actor(*args, **kwargs):
        if "controller" not in registry:
            raise ValueError("no controller")
        return registry["controller"]

    controller.options.return_value.remote.side_effect = create_controller

    def kill_controller(actor_handle, *, no_restart):
        assert no_restart is True
        if actor_handle is handle:
            registry.pop("controller", None)

    monkeypatch.setattr(interface.ray, "kill", MagicMock(side_effect=kill_controller))
    handle.store_config.remote.side_effect = lambda conf: setattr(handle.get_config.remote, "return_value", conf)
    monkeypatch.setattr(interface, "TransferQueueController", controller)
    monkeypatch.setattr(interface.ray, "get_actor", get_actor)
    monkeypatch.setattr(interface.ray, "get", lambda value: value)
    monkeypatch.setattr(interface, "_maybe_create_tq_client", MagicMock())
    monkeypatch.setattr(interface, "process_zmq_server_info", lambda _: {})
    monkeypatch.setattr(interface.time, "sleep", lambda _: pytest.fail("initialization waits for unpublished config"))
    storage = _mock_storage_initialization(monkeypatch)
    return controller, storage


def test_missing_storage_resource_leaves_no_controller_and_can_retry(monkeypatch):
    controller, storage = _mock_full_initialization(monkeypatch)
    nodes = [_node(_NODE_A, resources={"CPU": 4, "control_pool": 1})]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)
    conf = _simple_storage_conf("storage_pool")
    with pytest.raises(ValueError, match="storage_pool"):
        interface.init(conf)
    controller.options.assert_not_called()
    storage.options.assert_not_called()
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None
    nodes[0]["Resources"]["storage_pool"] = 1
    interface.init(conf)
    assert controller.options.return_value.remote.call_count == 1
    assert storage.options.call_count == 2


@pytest.mark.parametrize("controller_resource", [_UNSET, None, "storage_pool"])
def test_storage_accounts_for_actual_controller_cpu(monkeypatch, controller_resource):
    controller, storage = _mock_full_initialization(monkeypatch)
    nodes = [
        {**_node(_NODE_A, resources={"CPU": 1, "storage_pool": 1}), "NodeManagerAddress": "10.0.0.1"},
        {**_node(_NODE_B, resources={"CPU": 2, "storage_pool": 1}), "NodeManagerAddress": "10.0.0.2"},
    ]
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)
    conf = _simple_storage_conf("storage_pool")
    if controller_resource is not _UNSET:
        conf.controller = {"required_node_resource": controller_resource}
    interface.init(conf)
    if controller_resource in (_UNSET, None):
        controller.options.assert_called_once_with(name="TransferQueueController", namespace="transfer_queue")
    assert [c.kwargs["scheduling_strategy"].node_id for c in storage.options.call_args_list] == [_NODE_B, _NODE_B]


@pytest.mark.parametrize("after_preflight", ["missing", "controller_occupancy"])
def test_storage_recheck_failure_cleans_owned_controller_and_can_retry(monkeypatch, after_preflight):
    controller, storage = _mock_full_initialization(monkeypatch)
    nodes = [_node(_NODE_A, resources={"CPU": 2, "storage_pool": 1})]
    snapshots = MagicMock(side_effect=[nodes, [] if after_preflight == "missing" else nodes])
    monkeypatch.setattr(common.ray, "nodes", snapshots)
    conf = _simple_storage_conf("storage_pool")
    with pytest.raises(ValueError):
        interface.init(conf)
    handle = controller.options.return_value.remote.return_value
    interface.ray.kill.assert_called_once_with(handle, no_restart=True)
    handle.store_config.remote.assert_not_called()
    storage.options.assert_not_called()
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None
    nodes[0]["Resources"]["CPU"] = 3
    monkeypatch.setattr(common.ray, "nodes", lambda: nodes)
    interface.init(conf)
    assert controller.options.return_value.remote.call_count == 2
    assert storage.options.call_count == 2
    assert handle.store_config.remote.call_count == 1


def test_existing_controller_reuse_does_not_validate_or_kill(monkeypatch):
    controller, storage = _mock_full_initialization(monkeypatch)
    handle = controller.options.return_value.remote.return_value
    interface._TQ_CONTROLLER = handle
    handle.get_config.remote.return_value = OmegaConf.create({})
    nodes = MagicMock(side_effect=AssertionError("reuse must not replan"))
    monkeypatch.setattr(common.ray, "nodes", nodes)
    interface.init(_simple_storage_conf("missing_pool"))
    controller.options.assert_not_called()
    storage.options.assert_not_called()
    interface.ray.kill.assert_not_called()


@pytest.mark.parametrize("cpu", [0, 0.5])
def test_affinity_rejects_nodes_without_one_cpu_slot(monkeypatch, cpu):
    monkeypatch.setattr(common.ray, "nodes", lambda: [_node(_NODE_A, resources={"CPU": cpu, "control_pool": 1})])
    with pytest.raises(ValueError, match="CPU"):
        common.get_node_round_robin_scheduling_strategies(1, "control_pool")


@pytest.mark.parametrize("error_type", [ActorUnschedulableError, RayActorError])
def test_failed_controller_start_can_retry(monkeypatch, error_type):
    controller, storage = _mock_full_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [_node(_NODE_A, resources={"control_pool": 1})])
    server_info = MagicMock(side_effect=error_type("node disappeared"))
    monkeypatch.setattr(interface, "process_zmq_server_info", server_info)
    conf = OmegaConf.create({"controller": {"required_node_resource": "control_pool"}})
    with pytest.raises(error_type):
        interface.init(conf)
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None
    storage.options.assert_not_called()
    interface.ray.kill.assert_called_once_with(controller.options.return_value.remote.return_value, no_restart=True)
    server_info.side_effect = None
    server_info.return_value = {}
    monkeypatch.setattr(simple_storage_bootstrap, "get_placement_group", lambda *args, **kwargs: MagicMock())
    interface.init(conf)
    assert controller.options.return_value.remote.call_count == 2
    assert storage.options.call_count == 2


@pytest.mark.parametrize("error_type", [ActorUnschedulableError, RayActorError])
def test_attacher_discards_dead_controller_without_killing_it(monkeypatch, error_type):
    controller, storage = _mock_full_initialization(monkeypatch)
    dead = MagicMock()
    dead.get_config.remote.side_effect = error_type("initializer rolled back")
    interface._TQ_CONTROLLER = dead
    with pytest.raises(error_type):
        interface.init()
    assert interface._TQ_CONTROLLER is None
    interface.ray.kill.assert_not_called()
    replacement = MagicMock()
    replacement.get_config.remote.return_value = OmegaConf.create({})
    lookup = MagicMock(return_value=replacement)
    monkeypatch.setattr(interface.ray, "get_actor", lookup)
    interface.init()
    lookup.assert_called_once_with("TransferQueueController", namespace="transfer_queue")
    assert interface._TQ_CONTROLLER is replacement
    controller.options.assert_not_called()
    storage.options.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["storage_start", "store_config"])
def test_initialization_actor_failure_removes_only_owned_actors(monkeypatch, failure_stage):
    controller, storage = _mock_full_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [_node(_NODE_A, resources={"storage_pool": 1})])
    storage_handles = [MagicMock(), MagicMock()]
    storage.options.return_value.remote.side_effect = storage_handles
    controller_handle = controller.options.return_value.remote.return_value
    if failure_stage == "storage_start":
        monkeypatch.setattr(
            simple_storage_bootstrap,
            "process_zmq_server_info",
            MagicMock(side_effect=ActorUnschedulableError("node disappeared")),
        )
    else:
        controller_handle.store_config.remote.side_effect = RayActorError("controller died")
    with pytest.raises((ActorUnschedulableError, RayActorError)):
        interface.init(_simple_storage_conf("storage_pool"))
    assert interface.ray.kill.call_args_list == [
        call(storage_handles[1], no_restart=True),
        call(storage_handles[0], no_restart=True),
        call(controller_handle, no_restart=True),
    ]
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None


def test_racing_controller_creator_does_not_kill_winner(monkeypatch):
    controller, storage = _mock_full_initialization(monkeypatch)
    existing = MagicMock()
    existing.get_config.remote.return_value = OmegaConf.create({})
    monkeypatch.setattr(interface.ray, "get_actor", MagicMock(side_effect=[ValueError("not found"), existing]))
    controller.options.return_value.remote.side_effect = ValueError("name already exists")
    monkeypatch.setattr(common.ray, "nodes", lambda: [_node(_NODE_A, resources={"storage_pool": 1})])
    interface.init(_simple_storage_conf("storage_pool"))
    assert interface._TQ_CONTROLLER is existing
    storage.options.assert_not_called()
    interface.ray.kill.assert_not_called()


@pytest.mark.parametrize("storage_resource", [_UNSET, None, "storage_pool"])
@pytest.mark.parametrize("failure_stage", ["second_actor", "storage_ready", "store_config"])
def test_controller_affinity_rolls_back_storage_and_placement_group(monkeypatch, storage_resource, failure_stage):
    controller, storage = _mock_full_initialization(monkeypatch)
    monkeypatch.setattr(common.ray, "nodes", lambda: [_node(_NODE_A, resources={"control_pool": 1, "storage_pool": 1})])
    placement_group = MagicMock()
    monkeypatch.setattr(simple_storage_bootstrap, "get_placement_group", lambda *args, **kwargs: placement_group)
    remove_pg = MagicMock()
    monkeypatch.setattr(interface.ray.util, "remove_placement_group", remove_pg)
    handles = [MagicMock(), MagicMock()]
    controller_handle = controller.options.return_value.remote.return_value
    storage.options.return_value.remote.side_effect = handles
    if failure_stage == "second_actor":
        storage.options.return_value.remote.side_effect = [handles[0], ValueError("name belongs to another actor")]
    elif failure_stage == "storage_ready":
        monkeypatch.setattr(
            simple_storage_bootstrap, "process_zmq_server_info", MagicMock(side_effect=RayActorError("node lost"))
        )
    else:
        controller_handle.store_config.remote.side_effect = RayActorError("controller lost")
    conf = _simple_storage_conf(storage_resource)
    conf.controller = {"required_node_resource": "control_pool"}
    with pytest.raises((ValueError, RayActorError)):
        interface.init(conf)
    created_handles = handles[:1] if failure_stage == "second_actor" else handles
    assert interface.ray.kill.call_args_list == [
        *[call(handle, no_restart=True) for handle in reversed(created_handles)],
        call(controller_handle, no_restart=True),
    ]
    if storage_resource in (_UNSET, None):
        remove_pg.assert_called_once_with(placement_group)
    else:
        remove_pg.assert_not_called()
    assert interface._TQ_CONTROLLER is None
    assert interface._TQ_STORAGE is None

    storage.options.return_value.remote.side_effect = [MagicMock(), MagicMock()]
    monkeypatch.setattr(simple_storage_bootstrap, "process_zmq_server_info", lambda _: {})
    controller_handle.store_config.remote.side_effect = None
    interface.init(conf)
    assert controller.options.return_value.remote.call_count == 2
    # Successful initialization must not execute the rollback callbacks.
    assert interface.ray.kill.call_count == len(created_handles) + 1
