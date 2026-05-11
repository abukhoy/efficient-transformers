# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from QEfficient.generation.text_generation_inference import write_io_files
from QEfficient.utils import constants

logger = logging.getLogger(__name__)


class _SingleOutputModule(nn.Module):
    """Wrap a module whose forward may return tuples and expose one tensor output."""

    def __init__(self, module: nn.Module, output_index: int = 0) -> None:
        super().__init__()
        self.module = module
        self.output_index = output_index

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        outputs = self.module(hidden_states)
        return outputs[self.output_index] if isinstance(outputs, tuple) else outputs


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("-", "_")


def _first_parameter_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    return torch.float32


def _export_module_onnx(
    module: nn.Module,
    onnx_path: Path,
    hidden_size: int,
    sequence_length: int,
    output_name: str,
    dtype: torch.dtype,
) -> None:
    dummy_hidden_states = torch.zeros((1, sequence_length, hidden_size), dtype=dtype)
    torch.onnx.export(
        module,
        (dummy_hidden_states,),
        str(onnx_path),
        input_names=["hidden_states"],
        output_names=[output_name],
        dynamic_axes={
            "hidden_states": {0: "batch_size", 1: "seq_len"},
            output_name: {0: "batch_size", 1: "seq_len"},
        },
        opset_version=constants.ONNX_EXPORT_OPSET,
    )


def _write_module_io(
    module: nn.Module,
    io_dir: Path,
    hidden_size: int,
    prefill_sequence_length: int,
    decode_sequence_length: int,
    output_name: str,
    dtype: torch.dtype,
) -> None:
    phases = (
        ("prefill", prefill_sequence_length, True),
        ("decode", decode_sequence_length, False),
    )

    with torch.no_grad():
        for phase_name, sequence_length, reset in phases:
            hidden_states = torch.zeros((1, sequence_length, hidden_size), dtype=dtype)
            output = module(hidden_states)
            inputs = {"hidden_states": hidden_states.detach().cpu().numpy()}
            outputs = {output_name: output.detach().cpu().numpy()}
            write_io_files(
                inputs=inputs,
                outputs=outputs,
                write_io_dir=str(io_dir),
                write_io_subdir=phase_name,
                write_io_name="aic_batch_io",
                include_dims=True,
                reset=reset,
            )


def _get_benchmark_modules(model: nn.Module) -> List[Dict[str, object]]:
    get_benchmark_modules = getattr(model, "get_benchmark_modules", None)
    if callable(get_benchmark_modules):
        return get_benchmark_modules()

    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        return []

    modules = []
    for index, block in enumerate(blocks):
        if hasattr(block, "attn"):
            modules.append({"name": f"transformer.h.{index}.attn", "type": "attention", "module": block.attn})
        if hasattr(block, "mlp"):
            modules.append({"name": f"transformer.h.{index}.mlp", "type": "mlp", "module": block.mlp})
    return modules


def export_benchmark_modules(
    qeff_model,
    export_dir: Path,
    example_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Optional[Path]:
    """Export standalone benchmark ONNX and IO files for supported internal modules."""
    if getattr(qeff_model, "_is_weights_offloaded", False):
        raise RuntimeError(
            "Cannot export benchmark modules after PyTorch weights have been offloaded. "
            "Reload the model with from_pretrained(..., enable_benchmark=True) before export or compile."
        )

    modules = _get_benchmark_modules(qeff_model.model)
    if not modules:
        logger.warning("enable_benchmark=True was requested, but no benchmark modules were found for this model.")
        return None

    input_ids = (example_inputs or {}).get("input_ids")
    prefill_sequence_length = (
        int(input_ids.shape[1]) if input_ids is not None else constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN
    )
    decode_sequence_length = 1
    hidden_size = getattr(qeff_model.model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(qeff_model.model.config, "n_embd")
    hidden_size = int(hidden_size)

    benchmark_dir = export_dir / "benchmark_modules"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model_name": qeff_model.model_name,
        "prefill_sequence_length": prefill_sequence_length,
        "decode_sequence_length": decode_sequence_length,
        "hidden_size": hidden_size,
        "modules": [],
    }

    for module_info in modules:
        module = module_info["module"]
        module_name = str(module_info["name"])
        module_type = str(module_info["type"])
        output_name = f"{module_type}_output"
        module_dir = benchmark_dir / _safe_name(module_name)
        module_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = module_dir / "model.onnx"
        io_dir = module_dir / "io_dir"

        wrapped_module = _SingleOutputModule(module).eval()
        dtype = _first_parameter_dtype(module)

        if not onnx_path.is_file():
            _export_module_onnx(
                module=wrapped_module,
                onnx_path=onnx_path,
                hidden_size=hidden_size,
                sequence_length=prefill_sequence_length,
                output_name=output_name,
                dtype=dtype,
            )

        _write_module_io(
            module=wrapped_module,
            io_dir=io_dir,
            hidden_size=hidden_size,
            prefill_sequence_length=prefill_sequence_length,
            decode_sequence_length=decode_sequence_length,
            output_name=output_name,
            dtype=dtype,
        )

        manifest["modules"].append(
            {
                "name": module_name,
                "type": module_type,
                "onnx_path": str(onnx_path),
                "io_json_path": str(io_dir / "aic_batch_io.json"),
            }
        )

    manifest_path = benchmark_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Benchmark modules exported to %s", benchmark_dir)
    return benchmark_dir
