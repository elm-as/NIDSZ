"""Single-batch inference throughput and latency benchmarking for SOC deployment."""

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np
import pandas as pd

from src.core.contracts import LatencyBenchmark


@dataclass(frozen=True)
class LatencyProfileRequest:
    """Parameters governing a single-batch inference timing run."""

    model_name: str
    feature_mode: str
    batch_size: int
    repeats: int


def benchmark_batch_inference(
    score_fn: Callable[[pd.DataFrame], np.ndarray],
    x_sample: pd.DataFrame,
    request: LatencyProfileRequest,
) -> LatencyBenchmark:
    """Measure median batch latency, flows/sec throughput, and us/flow cost."""
    effective_size = min(request.batch_size, len(x_sample))
    if effective_size <= 0:
        raise ValueError("Cannot benchmark inference on an empty feature batch.")

    batch = x_sample.iloc[:effective_size]
    # Warm-up pass absorbs JIT/cache initialization before timing repeats.
    _ = score_fn(batch)

    elapsed_sec: list[float] = []
    for _ in range(request.repeats):
        start = perf_counter()
        _ = score_fn(batch)
        elapsed_sec.append(perf_counter() - start)

    median_sec = float(np.median(elapsed_sec))
    median_ms = median_sec * 1_000.0
    throughput = float(effective_size / max(median_sec, 1e-9))
    us_per_flow = float((median_sec * 1_000_000.0) / effective_size)
    return LatencyBenchmark(
        model_name=request.model_name,
        feature_mode=request.feature_mode,
        batch_size=int(effective_size),
        median_ms=median_ms,
        throughput_flows_per_sec=throughput,
        latency_us_per_flow=us_per_flow,
    )
