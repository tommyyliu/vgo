"""Queue-driven reinforcement-learning pipeline entry point.

The implementation lives in :mod:`vgo_training.pipeline`.  This module keeps
the historical command name while deliberately exposing only the new
continuous actor/replay/learner coordinator.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import signal
from typing import Sequence

from .pipeline import (
    Pipeline,
    PipelineConfig,
    add_pipeline_arguments,
    atomic_json,
    cargo_executable,
    config_from_arguments,
    runtime_environment,
)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pipelined VGO actor/replay/learner loop. Replay shards and "
            "model publications are immutable and crash-recoverable; Elo work "
            "is queued off the learning critical path."
        )
    )
    add_pipeline_arguments(parser)
    parser.add_argument(
        "--drain-telemetry",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "after the learning pipeline completes, run queued Elo matches; "
            "this is an operational choice and is not part of the run identity"
        ),
    )
    parser.add_argument(
        "--telemetry-only",
        action="store_true",
        help=(
            "load the existing configuration under --output and drain only its "
            "queued Elo jobs"
        ),
    )
    return parser.parse_args(argv)


def run(arguments: argparse.Namespace | PipelineConfig) -> dict[str, object]:
    if isinstance(arguments, PipelineConfig):
        return asyncio.run(Pipeline(arguments).run())

    pipeline = (
        Pipeline.resume(arguments.output)
        if arguments.telemetry_only
        else Pipeline(config_from_arguments(arguments))
    )

    async def execute() -> dict[str, object]:
        if arguments.telemetry_only:
            await pipeline.drain_telemetry()
            return pipeline.report()
        report = await pipeline.run()
        if arguments.drain_telemetry:
            await pipeline.drain_telemetry()
            report = pipeline.report()
        return report

    return asyncio.run(execute())


def _terminate_like_interrupt(signum: int, frame: object) -> None:
    raise KeyboardInterrupt


def main(argv: Sequence[str] | None = None) -> None:
    # Route SIGTERM onto the Ctrl-C path, which already works. Without this the
    # default disposition kills the interpreter outright, `_run`'s finally never
    # runs, and two things follow: the generation and arena children -- spawned
    # with start_new_session=True so they survive their parent -- are orphaned
    # rather than signalled, and the session's wall time is never accumulated.
    # SIGTERM is how a detached run is stopped by hand and how a provider asks
    # an instance to shut down, so it is the common case, not the rare one.
    #
    # Installed here rather than in Pipeline: a library that rewrites signal
    # handlers on import surprises everything that embeds it, including the
    # tests.
    signal.signal(signal.SIGTERM, _terminate_like_interrupt)
    report = run(parse_arguments(argv))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
