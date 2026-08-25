"""Typed configuration loading.

One rule: every tunable lives in configs/*.toml; machine-local overrides come
from SAHEL_<SECTION>_<KEY> environment variables; per-invocation overrides come
from CLI flags. Unknown keys are a hard error (mirrors the competition
schema's additionalProperties: false discipline).
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict


def repo_root() -> Path:
    """The repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def _expand(value: str) -> Path:
    return Path(os.path.expandvars(value)).expanduser()


class LabPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str
    audit_bench: str
    audit_server: str
    audit_cli: str
    native_quantize: str
    native_imatrix: str
    convert_script: str
    profiler_python: str
    models_dir: str

    def path(self, name: str) -> Path:
        return _expand(getattr(self, name))


class BenchCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    taskset_cores: str = "0-3"
    threads: int = 4


class RepoCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_path: str
    library_db: str

    def abs_model_path(self) -> Path:
        return repo_root() / self.model_path

    def abs_library_db(self) -> Path:
        return repo_root() / self.library_db


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lab: LabPaths
    bench: BenchCfg
    repo: RepoCfg


class RetrievalCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    k: int = 4
    max_per_doc: int = 2
    rrf_k: int = 60


class ConfidenceCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    threshold: float


class RetrievalSettings(BaseModel):
    """configs/retrieval.toml, the calibrated abstention threshold lives here.

    This file existed and was cited in REPORT.md for two days before anything
    read it, so the app silently ran on the uncalibrated default instead. There
    is no default for `threshold`: a missing value must fail loudly rather than
    quietly restore that situation.
    """

    model_config = ConfigDict(extra="forbid")

    retrieval: RetrievalCfg = RetrievalCfg()
    confidence: ConfidenceCfg


def _apply_env_overrides(data: dict) -> dict:
    """SAHEL_<SECTION>_<KEY> environment variables override toml values."""
    for section, values in data.items():
        if not isinstance(values, dict):
            continue
        for key in values:
            env_name = f"SAHEL_{section.upper()}_{key.upper()}"
            if env_name in os.environ:
                values[key] = os.environ[env_name]
    return data


def load_settings(path: Path | None = None) -> Settings:
    path = path or repo_root() / "configs" / "settings.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Settings.model_validate(_apply_env_overrides(data))


def load_retrieval(path: Path | None = None) -> RetrievalSettings:
    path = path or repo_root() / "configs" / "retrieval.toml"
    with path.open("rb") as f:
        data = tomllib.load(f)
    return RetrievalSettings.model_validate(_apply_env_overrides(data))
