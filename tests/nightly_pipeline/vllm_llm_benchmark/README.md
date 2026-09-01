# vLLM LLM Benchmark Pipeline

This directory contains the vLLM QAIC benchmark runners used by
`scripts/JenkinsfileVllmLlmBenchmark`. There are four runner scripts, each
covering one model domain, sharing all CSV parsing, command building, and
subprocess orchestration from `vllm_benchmark_common.py`:

- `vllm_llm_benchmark.py` — LLM serving configs.
- `vllm_embedding_benchmark.py` — embedding-model serving configs.
- `vllm_audio_benchmark.py` — audio (whisper) serving configs.
- `vllm_vlm_benchmark.py` — vision-language model serving configs.

Each script defines its own `LATEST_MODELS` curated set; `SKIPPED_MODELS`
(the >70B hardware-capability exclusion) is shared and applied to all four.

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
| VLM | `configs/vlm_configs.csv` | `vlm_results.csv` |

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

python3 tests/nightly_pipeline/vllm_llm_benchmark/vllm_vlm_benchmark.py \
  --config-name vlm \
  --input-csv tests/nightly_pipeline/vllm_llm_benchmark/configs/vlm_configs.csv \
  --output-csv /tmp/vllm_vlm_smoke/vlm_results.csv \
  --results-dir /tmp/vllm_vlm_smoke \
  --rows 1 \
  --dry-run
```

## Jenkins Flow

The Jenkins pipeline:

1. Creates or reuses a Python venv.
2. Clones `https://github.com/qualcomm/vllm-qaic.git`.
3. Runs `scripts/install.sh aot`, which installs QEfficient internally.
4. Clones and installs `qaic-disagg`.
5. Runs the selected LLM/embedding/audio/VLM config CSVs.
6. Archives `vllm_llm_results/**/*.csv` and `vllm_llm_results/**/*.log`.

Use `DRY_RUN=true` and `ROWS_DEFAULT=1` to verify the gpt2 command and output
CSV/log generation on a Jenkins agent without consuming QAIC runtime. The
embedding, audio, and VLM stages have their own `ROWS_EMBEDDING`/`ROWS_AUDIO`/
`ROWS_VLM` and `RUN_EMBEDDING`/`RUN_AUDIO`/`RUN_VLM` params for the same
purpose.

The VLM stage additionally exposes three `choice()` params that filter rows
within `configs/vlm_configs.csv` (rather than selecting a different CSV/stage
per combination):

- `VLM_DISAGG_MODE` (`ALL`/`ED`/`PD`/`EPD`) — filters by the `disagg_mode`
  column: encode-decode, prefill-decode (no vision/encode stage), or
  encode-prefill-decode.
- `VLM_SPECIALIZATION_MODE` (`ALL`/`single`/`multi`) — filters by the
  `specialization_mode` column: single vs multi resolution/image-count
  buckets on the client's `--random-mm-bucket-config`.
- `VLM_BLOCKING_MODE` (`ALL`/`blocking`/`non_blocking`) — filters by the
  `blocking_mode` column.

These map to the runner's own `--disagg-mode`/`--specialization-mode`/
`--blocking-mode` CLI flags (also `ALL` by default), which no-op for any CSV
row lacking the corresponding column — so the LLM/embedding/audio CSVs are
unaffected.

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
`prefill_override_qaic_config`, and `decode_override_qaic_config`. VLM rows
additionally use the encode-stage columns (`encode_port`,
`encode_device_group`, `encode_max_num_seqs`, `encode_override_qaic_config`)
for EPD layouts; ED rows leave the prefill columns blank, and PD rows leave
the encode columns blank — `qaic_disagg`'s CLI flags for the omitted stage are
simply not passed.

The embedding CSV (`configs/embedding_configs.csv`) has four rows per model,
one per `pooling_method` (`mean`, `avg`, `cls`, `max`), passed through the
`additional_config` column's `override_qaic_config.pooling_method` key, along
with the qserve-only client columns `tokenizer` and `random_range_ratio`.

The audio CSV (`configs/audio_configs.csv`) has one row per whisper model and
uses the qserve-only client columns `dataset_path`, `hf_subset`, `hf_split`,
and `hf_output_len` for the `hf` dataset backend.

The VLM CSV (`configs/vlm_configs.csv`) has one row per (model, disagg_mode,
specialization_mode, blocking_mode) combination. Rows tagged
`*_verified` reproduce a user-supplied example command token-for-token
(modulo host/port/paths); rows tagged `*_inferred_unverified` are best-effort
constructions for combinations no example was supplied for, and should be
validated against real hardware output before being treated as a golden
reference. Multi-specialization rows set `random_mm_limit_mm_per_prompt`,
`random_mm_bucket_config` (a Python-tuple-keyed literal, not JSON — passed
through to `vllm bench serve --random-mm-bucket-config` verbatim), and the
server-side `limit_mm_per_prompt` JSON column. Blocking rows embed a nested
`qaic_config":{"enable_blocking":true,...}` block inside the relevant
`*_override_qaic_config` column, the same mechanism used by the existing LLM
blocking CSV.

## Model Filtering

After `--rows` selects which CSV rows are in play, two model-name filters are
applied, in order:

1. **`SKIPPED_MODELS`** (always enforced, no override): a hardcoded set of
   models too large to run in this pipeline (>70B real params) — currently
   `zai-org/GLM-4.5` (~355B, MoE) and `hpcai-tech/grok-1` (~314B).
2. **`LATEST_MODELS_ONLY`** (`--latest-models-only`, default `true`): when
   enabled, restricts runs to the curated `LATEST_MODELS` set defined in the
   runner script for the domain being run (`vllm_llm_benchmark.py`,
   `vllm_embedding_benchmark.py`, `vllm_audio_benchmark.py`, or
   `vllm_vlm_benchmark.py`). Set to `false` (Jenkins param `LATEST_MODELS_ONLY`)
   to run all non-skipped models in the CSV.

`zai-org/GLM-4.5` is in both LLM's `LATEST_MODELS` and `SKIPPED_MODELS` — the
skip-list always wins, so it is excluded regardless of the latest-only flag.

For the VLM runner specifically, three additional filters apply after the two
above: `--disagg-mode`, `--specialization-mode`, and `--blocking-mode` (each
default `ALL`), matching the `disagg_mode`/`specialization_mode`/
`blocking_mode` CSV columns. They no-op for rows/CSVs lacking the column, so
they have no effect on the LLM/embedding/audio runners.

Skipped rows are logged to stdout with the reason and do not appear in the
output CSV, matching how `enabled=false` rows are already excluded silently.


