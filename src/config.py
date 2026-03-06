from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional


SubsetType = Literal["state_store_cat", "random_series"]


@dataclass
class PathsConfig:
    """Configuration for project paths."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"


@dataclass
class SubsetConfig:
    """Configuration controlling which series are used for training."""

    subset_type: SubsetType = "state_store_cat"

    # Option A: 1 state + 1 store + 1 category
    state_id: str = "CA"
    store_id: str = "CA_1"
    cat_id: str = "HOBBIES"

    # Option B: random N series
    n_random_series: int = 200
    random_seed: int = 42


@dataclass
class SplitConfig:
    """Configuration for train/validation split."""

    validation_days: int = 56


@dataclass
class ModelConfig:
    """Configuration for the XGBoost model and training."""

    # XGBoost core parameters
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 8
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    reg_alpha: float = 0.0
    min_child_weight: float = 1.0

    n_jobs: int = 4
    early_stopping_rounds: int = 50

    # Allow overriding arbitrary params if needed
    extra_params: Dict[str, object] = field(default_factory=dict)


@dataclass
class ForecastConfig:
    """Configuration for forecasting."""

    forecast_days: int = 28


@dataclass
class ProjectConfig:
    """Top-level configuration container."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    subset: SubsetConfig = field(default_factory=SubsetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)


def get_default_config() -> ProjectConfig:
    """Return a default `ProjectConfig` instance."""

    return ProjectConfig()


def ensure_project_dirs(paths: Optional[PathsConfig] = None) -> None:
    """Ensure models and outputs directories exist."""

    paths = paths or PathsConfig()
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    paths.outputs_dir.mkdir(parents=True, exist_ok=True)

