#!/usr/bin/env python3
"""
Run CSV-driven vision-language model serving benchmarks through vLLM QAIC.

Each CSV row describes one VLM run for one of three disagg layouts
(disagg_mode column: ED, PD, EPD) at one specialization_mode (single/multi
resolution-image-count buckets) and one blocking_mode (blocking/non_blocking).
Shared CSV parsing, command building, and subprocess orchestration live in
vllm_benchmark_common.py, shared with the LLM, embedding, and audio runners.
"""

from __future__ import annotations

from vllm_benchmark_common import build_arg_parser, run_benchmarks

LATEST_MODELS = {
    "google/gemma-4-26B-A4B-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "google/gemma-4-31B-it",
}


def main() -> int:
    args = build_arg_parser("Run VLM vLLM QAIC benchmark CSV rows.").parse_args()
    return run_benchmarks(args, LATEST_MODELS)


if __name__ == "__main__":
    raise SystemExit(main())
