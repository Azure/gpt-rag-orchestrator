"""Non-gating microbenchmark for enabled metadata-only audit emission."""

from __future__ import annotations

import logging
import statistics
import sys
import time
import tracemalloc

from telemetry.audit import AuditEmitter
from telemetry.audit_contract import (
    AuditSettings,
    AuditStatus,
    EventType,
    ReasonCode,
)


def main(iterations: int = 10_000) -> None:
    logger = logging.getLogger("gptrag.audit")
    logger.handlers = [logging.NullHandler()]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    emitter = AuditEmitter(
        AuditSettings(
            enabled=True,
            sensitive_content_enabled=False,
            sensitive_content_fields=frozenset(),
            actor_pseudonym_enabled=False,
            source_event_limit=25,
            hmac_key_id="v1",
            hmac_key=b"k" * 32,
            additional_redacted_keys=frozenset(),
        ),
        service_name="gpt-rag-orchestrator",
        service_version="benchmark",
        environment="benchmark",
    )

    samples = []
    tracemalloc.start()
    for _ in range(iterations):
        started = time.perf_counter_ns()
        emitter.emit(
            EventType.ROUTE_SELECTED,
            operation="benchmark",
            status=AuditStatus.SELECTED,
            reason_code=ReasonCode.STRATEGY_CONFIGURED,
            metadata={
                "decision_type": "agent_strategy",
                "decision_value": "maf_lite",
            },
        )
        samples.append((time.perf_counter_ns() - started) / 1_000)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    samples.sort()
    p95 = samples[int(len(samples) * 0.95)]
    sys.stdout.write(
        "audit metadata emission: "
        f"median={statistics.median(samples):.2f}us "
        f"p95={p95:.2f}us peak_memory={peak / 1024:.1f}KiB "
        f"iterations={iterations}\n"
    )


if __name__ == "__main__":
    main()
