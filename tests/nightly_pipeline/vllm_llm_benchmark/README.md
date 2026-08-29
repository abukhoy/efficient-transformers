# vLLM LLM Benchmark Pipeline

This directory contains the LLM-only vLLM QAIC benchmark runner used by
`scripts/JenkinsfileVllmLlmBenchmark`.

The runner is CSV-driven. Each input CSV owns one serving mode and produces one
output CSV:

| Mode | Input CSV | Output CSV |
| --- | --- | --- |
| default CB + subfunction | `configs/llm_default_configs.csv` | `llm_default_results.csv` |
| CCL enabled | `configs/llm_ccl_configs.csv` | `llm_ccl_results.csv` |
| blocking | `configs/llm_blocking_configs.csv` | `llm_blocking_results.csv` |
| disagg PD | `configs/llm_disagg_pd_configs.csv` | `llm_disagg_pd_results.csv` |

## Local Dry Run

Dry run validates CSV parsing, command generation, log paths, and output CSV
writing without starting vLLM or using QAIC devices:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_llm_benchmark.py \
  --config-name default \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/llm_default_configs.csv \
  --output-csv /tmp/vllm_llm_smoke/llm_default_results.csv \
  --results-dir /tmp/vllm_llm_smoke \
  --rows 1 \
  --dry-run
```

Row 1 in the default CSV is `openai-community/gpt2`, intended for a quick smoke
check before running all rows.

## Jenkins Flow

The Jenkins pipeline:

1. Creates or reuses a Python venv.
2. Clones `https://github.com/qualcomm/vllm-qaic.git`.
3. Runs `scripts/install.sh aot`, which installs QEfficient internally.
4. Clones and installs `qaic-disagg`.
5. Runs the selected LLM config CSVs.
6. Archives `vllm_llm_results/**/*.csv` and `vllm_llm_results/**/*.log`.

Use `DRY_RUN=true` and `ROWS_DEFAULT=1` to verify the gpt2 command and output
CSV/log generation on a Jenkins agent without consuming QAIC runtime.

## CSV Notes

Common columns:

- `model`, `tag`, `server_type`, `client_type`, `host`, `port`
- `PL`, `GL`, `CL`, `num_prompts`, `max_concurrency`
- `backend`, `endpoint`, `dataset_name`, `ignore_eos`
- `server_extra_args`, `client_extra_args` for mode-specific flags not yet
  modeled as first-class columns

For `server_type=api_server`, the runner builds
`python -m vllm.entrypoints.openai.api_server`. The `additional_config` JSON is
constructed from columns such as `device_group`, `use_onnx_subfunctions`,
`ccl_enabled`, `comp_ctx_lengths_prefill`, `comp_ctx_lengths_decode`, and the
blocking columns.

For `server_type=qaic_disagg`, the runner builds `python -m qaic_disagg` using
the PD columns such as `prefill_device_group`, `decode_device_group`,
`prefill_override_qaic_config`, and `decode_override_qaic_config`.

## Model Filtering

After `--rows` selects which CSV rows are in play, two model-name filters are
applied, in order:

1. **`SKIPPED_MODELS`** (always enforced, no override): a hardcoded set of
   models too large to run in this pipeline (>70B real params) — currently
   `zai-org/GLM-4.5` (~355B, MoE) and `hpcai-tech/grok-1` (~314B).
2. **`LATEST_MODELS_ONLY`** (`--latest-models-only`, default `true`): when
   enabled, restricts runs to the curated `LATEST_MODELS` set in
   `vllm_llm_benchmark.py`. Set to `false` (Jenkins param `LATEST_MODELS_ONLY`)
   to run all non-skipped models in the CSV.

`zai-org/GLM-4.5` is in both `LATEST_MODELS` and `SKIPPED_MODELS` — the
skip-list always wins, so it is excluded regardless of the latest-only flag.

Skipped rows are logged to stdout with the reason and do not appear in the
output CSV, matching how `enabled=false` rows are already excluded silently.

