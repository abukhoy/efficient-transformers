# vLLM LLM Benchmark Pipeline

This directory contains the vLLM QAIC benchmark runners used by
`scripts/JenkinsfileVllmLlmBenchmark`. There are three runner scripts, each
covering one model domain, sharing all CSV parsing, command building, and
subprocess orchestration from `vllm_benchmark_common.py`:

- `vllm_llm_benchmark.py` — LLM serving configs.
- `vllm_embedding_benchmark.py` — embedding-model serving configs.
- `vllm_audio_benchmark.py` — audio (whisper) serving configs.

Each script defines its own `LATEST_MODELS` curated set; `SKIPPED_MODELS`
(the >70B hardware-capability exclusion) is shared and applied to all three.

The runners are CSV-driven. Each input CSV owns one serving mode and produces
one output CSV:

| Mode | Input CSV | Output CSV |
| --- | --- | --- |
| default CB + subfunction | `configs/llm_default_configs.csv` | `llm_default_results.csv` |
| CCL enabled | `configs/llm_ccl_configs.csv` | `llm_ccl_results.csv` |
| blocking | `configs/llm_blocking_configs.csv` | `llm_blocking_results.csv` |
| disagg PD | `configs/llm_disagg_pd_configs.csv` | `llm_disagg_pd_results.csv` |
| embedding | `configs/embedding_configs.csv` | `embedding_results.csv` |
| audio | `configs/audio_configs.csv` | `audio_results.csv` |

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

The embedding and audio runners work the same way:

```bash
python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_embedding_benchmark.py \
  --config-name embedding \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/embedding_configs.csv \
  --output-csv /tmp/vllm_embedding_smoke/embedding_results.csv \
  --results-dir /tmp/vllm_embedding_smoke \
  --rows 1 \
  --dry-run

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_audio_benchmark.py \
  --config-name audio \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/audio_configs.csv \
  --output-csv /tmp/vllm_audio_smoke/audio_results.csv \
  --results-dir /tmp/vllm_audio_smoke \
  --rows 1 \
  --dry-run
```

## Jenkins Flow

The Jenkins pipeline:

1. Creates or reuses a Python venv.
2. Clones `https://github.com/qualcomm/vllm-qaic.git`.
3. Runs `scripts/install.sh aot`, which installs QEfficient internally.
4. Clones and installs `qaic-disagg`.
5. Runs the selected LLM/embedding/audio config CSVs.
6. Archives `vllm_llm_results/**/*.csv` and `vllm_llm_results/**/*.log`.

Use `DRY_RUN=true` and `ROWS_DEFAULT=1` to verify the gpt2 command and output
CSV/log generation on a Jenkins agent without consuming QAIC runtime. The
embedding and audio stages have their own `ROWS_EMBEDDING`/`ROWS_AUDIO` and
`RUN_EMBEDDING`/`RUN_AUDIO` params for the same purpose.

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

The embedding CSV (`configs/embedding_configs.csv`) has four rows per model,
one per `pooling_method` (`mean`, `avg`, `cls`, `max`), passed through the
`additional_config` column's `override_qaic_config.pooling_method` key, along
with the qserve-only client columns `tokenizer` and `random_range_ratio`.

The audio CSV (`configs/audio_configs.csv`) has one row per whisper model and
uses the qserve-only client columns `dataset_path`, `hf_subset`, `hf_split`,
and `hf_output_len` for the `hf` dataset backend.

## Model Filtering

After `--rows` selects which CSV rows are in play, two model-name filters are
applied, in order:

1. **`SKIPPED_MODELS`** (always enforced, no override): a hardcoded set of
   models too large to run in this pipeline (>70B real params) — currently
   `zai-org/GLM-4.5` (~355B, MoE) and `hpcai-tech/grok-1` (~314B).
2. **`LATEST_MODELS_ONLY`** (`--latest-models-only`, default `true`): when
   enabled, restricts runs to the curated `LATEST_MODELS` set defined in the
   runner script for the domain being run (`vllm_llm_benchmark.py`,
   `vllm_embedding_benchmark.py`, or `vllm_audio_benchmark.py`). Set to
   `false` (Jenkins param `LATEST_MODELS_ONLY`) to run all non-skipped models
   in the CSV.

`zai-org/GLM-4.5` is in both LLM's `LATEST_MODELS` and `SKIPPED_MODELS` — the
skip-list always wins, so it is excluded regardless of the latest-only flag.

Skipped rows are logged to stdout with the reason and do not appear in the
output CSV, matching how `enabled=false` rows are already excluded silently.


