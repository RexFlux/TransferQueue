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

"""Per-(loop, server, socket_name) ZMQ DEALER pool with per-call timeout and retry.

Background
==========
The original ``dynamic_socket`` / ``dynamic_storage_manager_socket`` decorators
created a fresh ``zmq.asyncio.Context``, opened a new DEALER, connected,
ran the method, then closed the socket and tore down the context for every
single API call. Under high-throughput async RL training this churned tens
of thousands of TIME_WAIT entries per minute at the controller's listen port,
exhausted the pod's ephemeral port range (32768-60999) and made the next
NCCL bootstrap ``bind(0)`` fail with EADDRINUSE.

This module replaces that pattern with a long-lived pool of DEALER sockets,
owned per client / storage manager instance, plus two pieces of defense
against silent ROUTER reply mis-routing:

1. ``asyncio.wait_for(timeout=...)`` around each call so a misrouted reply
   (or a stalled controller) cannot hang the recv indefinitely.
2. A retry loop that drops the suspect socket and tries a fresh one for
   the next attempt — typically the next request succeeds and the failure
   is invisible to the caller.

Identity construction (left to the caller via ``identity_prefix``) MUST
include both the local id (client_id / storage_manager_id) and the asyncio
loop id. Components that drive the same client instance from multiple loops
(e.g. one bg loop for sync wrappers + a shared loop for async calls) would
otherwise hand the SAME identity to one ROUTER from two different DEALERs,
which then routes replies non-deterministically between them.

``zmq.asyncio.Context`` is shared module-wide per asyncio loop because it
is loop-bound but otherwise process-wide; multiple managers on the same
loop share one context.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Awaitable, Callable, Optional

import zmq
import zmq.asyncio

from transfer_queue.utils.zmq_utils import create_zmq_socket

logger = logging.getLogger(__name__)


# Per-(loop, server, socket_name) DEALER pool size. 16 covers typical
# in-flight concurrency comfortably; raise via env on bursty workloads.
TQ_POOL_SIZE = max(1, int(os.environ.get("TQ_POOL_SIZE", "16")))

# Per-call asyncio-level timeout (seconds). Storage callers usually pass
# their own (larger) timeout to ``invoke_with_pool`` so this default is
# tuned to the controller's expected response latency.
TQ_REQUEST_TIMEOUT_S = float(os.environ.get("TQ_REQUEST_TIMEOUT_S", "30"))

# Maximum attempts per request. The first failure drops the socket from
# the pool; subsequent attempts acquire a fresh one. Set to 1 to disable
# retry entirely.
TQ_REQUEST_MAX_ATTEMPTS = max(1, int(os.environ.get("TQ_REQUEST_MAX_ATTEMPTS", "2")))


# zmq.asyncio.Context binds to a specific asyncio loop; keep one per loop,
# shared across all SocketPoolManager instances on that loop.
_CONTEXTS: dict[int, zmq.asyncio.Context] = {}


def _context_for_loop() -> zmq.asyncio.Context:
    loop_id = id(asyncio.get_running_loop())
    ctx = _CONTEXTS.get(loop_id)
    if ctx is None:
        ctx = zmq.asyncio.Context()
        _CONTEXTS[loop_id] = ctx
    return ctx


class PooledSocket:
    """Pool entry wrapping a DEALER socket with a broken-flag.

    When ``broken`` is set the socket is closed on release rather than
    returned to the free queue. The flag is set by ``invoke_with_pool``
    on any non-clean exit (timeout, exception, cancellation) because the
    socket's recv state is then unknown — a stale reply could otherwise
    poison the next user.
    """

    __slots__ = ("sock", "broken")

    def __init__(self, sock: zmq.asyncio.Socket):
        self.sock = sock
        self.broken = False


class SocketPool:
    """Async pool of long-lived DEALER sockets to one (server, socket_name).

    Grows lazily up to ``max_size``; further callers wait on the free queue.
    Each pooled socket serves one in-flight request at a time (enforced by
    the take/put discipline of the underlying ``asyncio.Queue``).
    """

    def __init__(
        self,
        *,
        context: zmq.asyncio.Context,
        address: str,
        ip: str,
        identity_prefix: str,
        max_size: int = TQ_POOL_SIZE,
        on_create: Optional[Callable[[zmq.asyncio.Socket], None]] = None,
    ):
        self._context = context
        self._address = address
        self._ip = ip
        self._identity_prefix = identity_prefix
        self._max_size = max_size
        self._on_create = on_create
        self._free: asyncio.Queue[PooledSocket] = asyncio.Queue()
        self._total = 0
        # Monotonic id for new socket identities; never decremented even
        # when a broken socket is dropped. This guarantees the ROUTER
        # never sees the same identity twice across the pool's lifetime,
        # which prevents stale routing-table entries from misdirecting
        # replies meant for a freshly-recreated slot.
        self._next_id = 0
        self._create_lock = asyncio.Lock()

    async def acquire(self) -> PooledSocket:
        """Check out a socket from the pool.

        Returns an idle socket if one is available. Otherwise, if the pool
        has not reached ``max_size``, creates a new DEALER socket with a
        unique identity, connects it, and returns it. If the pool is full,
        waits until a socket is released back.
        """
        try:
            return self._free.get_nowait()
        except asyncio.QueueEmpty:
            pass
        async with self._create_lock:
            if self._total < self._max_size:
                identity = f"{self._identity_prefix}-{self._next_id}".encode()
                self._next_id += 1
                sock = create_zmq_socket(
                    self._context,
                    zmq.DEALER,
                    ip=self._ip,
                    identity=identity,
                )
                sock.connect(self._address)
                if self._on_create is not None:
                    self._on_create(sock)
                self._total += 1
                return PooledSocket(sock)
        return await self._free.get()

    def release(self, ps: PooledSocket) -> None:
        """Return a socket to the pool.

        If the socket is marked broken, it is closed and dropped from the
        pool's total count so a fresh one can be created later. Otherwise it
        is placed back on the idle queue for reuse.
        """
        if ps.broken:
            try:
                if not ps.sock.closed:
                    ps.sock.close(linger=0)
            except Exception:
                pass
            self._total -= 1
            return
        self._free.put_nowait(ps)

    def drain_close(self) -> None:
        """Close every idle socket. Sockets currently checked out are
        left alone (their owner will close them on release-as-broken)."""
        while True:
            try:
                ps = self._free.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if not ps.sock.closed:
                    ps.sock.close(linger=0)
            except Exception:
                pass
            self._total -= 1


class SocketPoolManager:
    """Per-instance owner of ``SocketPool`` objects.

    Each ``AsyncTransferQueueClient`` / ``AsyncSimpleStorageManager``
    instance holds one of these. ``close()`` drains every pool the
    instance created, so resources are released when the owning client
    is disposed without waiting for process exit.

    Pool lookup is keyed by ``(current_loop_id, *pool_key)`` so the same
    instance driven from two asyncio loops keeps separate pools per loop
    (ZMQ sockets and asyncio.Queue/Lock are loop-bound).
    """

    def __init__(self) -> None:
        self._pools: dict[tuple, SocketPool] = {}

    def get_or_create(
        self,
        *,
        pool_key: tuple,
        address: str,
        ip: str,
        identity_prefix: str,
        on_create: Optional[Callable[[zmq.asyncio.Socket], None]] = None,
    ) -> SocketPool:
        """Look up or create the pool for ``(current_loop, *pool_key)``.

        Args:
            pool_key: Caller-supplied tuple uniquely identifying the
                (server, socket_name) combination. The current loop id
                is prepended internally so pools are loop-scoped.
            address: ``tcp://host:port`` style ZMQ address.
            ip: Server IP (used by ``create_zmq_socket`` to enable the
                IPv6 socket option when applicable).
            identity_prefix: Prefix for new DEALER identities. MUST
                include both the local id (client_id /
                storage_manager_id) and the current loop id; see the
                module docstring.
            on_create: Optional callback invoked with each freshly
                created socket, e.g. to set per-socket RCVTIMEO/SNDTIMEO.
        """
        loop_id = id(asyncio.get_running_loop())
        full_key = (loop_id, *pool_key)
        pool = self._pools.get(full_key)
        if pool is None:
            pool = SocketPool(
                context=_context_for_loop(),
                address=address,
                ip=ip,
                identity_prefix=identity_prefix,
                on_create=on_create,
            )
            self._pools[full_key] = pool
        return pool

    def close(self) -> None:
        """Drain and close every pool owned by this manager.

        Idempotent; safe to call multiple times. Sockets currently
        checked out by an in-flight call are left for that call to
        close (via ``broken=True`` on release).
        """
        for pool in self._pools.values():
            try:
                pool.drain_close()
            except Exception as e:
                logger.warning("Error draining socket pool: %s", e)
        self._pools.clear()


async def invoke_with_pool(
    pool: SocketPool,
    call: Callable[[zmq.asyncio.Socket], Awaitable],
    *,
    timeout: float = TQ_REQUEST_TIMEOUT_S,
    max_attempts: int = TQ_REQUEST_MAX_ATTEMPTS,
    label: str = "tq-call",
):
    """Acquire a pooled socket, run ``call(sock)`` with timeout and retry.

    The timeout guards against silent ROUTER reply mis-routing and
    controller stalls; retry transparently masks the resulting failure
    by dropping the suspect socket and acquiring a fresh one. The next
    request typically succeeds within the same logical call.

    Args:
        pool: Pool acquired from :meth:`SocketPoolManager.get_or_create`.
        call: Async callable invoked with the pooled socket; returns the
            awaitable that produces the method's result.
        timeout: Per-attempt asyncio-level timeout in seconds.
        max_attempts: Maximum number of tries before re-raising.
        label: Short tag used in retry log messages for context.

    Raises:
        Re-raises the last failure if all attempts fail, or
        ``asyncio.CancelledError`` immediately if the call is cancelled
        (cancellation is not retried).
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(1, max_attempts + 1):
        ps = await pool.acquire()
        clean_exit = False
        try:
            result = await asyncio.wait_for(call(ps.sock), timeout=timeout)
            clean_exit = True
            return result
        except asyncio.CancelledError:
            # Don't retry on cancel — the caller wants out. The finally
            # block still marks the socket broken because its recv state
            # is unknown. CancelledError is BaseException (Py3.8+), not
            # Exception, so a bare ``except Exception`` would miss it
            # and let a poisoned socket slip back into the pool.
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                logger.warning(
                    "[%s] attempt %d/%d failed (%s: %s); retrying with fresh socket",
                    label,
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    exc,
                )
        finally:
            if not clean_exit:
                ps.broken = True
            pool.release(ps)
    assert last_exc is not None
    raise last_exc
