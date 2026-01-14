"""ml4t-engineer - A Financial Machine Learning Feature Engineering Library.

ml4t-engineer is a comprehensive FML stack designed for correctness, reproducibility,
and performance. It provides tools for feature engineering, labeling, and validation
of financial machine learning models.

Agent Navigation:
    This package includes AGENT.md files for AI agent navigation.
    Call `get_agent_docs()` to get paths to all documentation files.
    Start with the root AGENT.md for package overview and navigation.
"""

from pathlib import Path as _Path

__version__ = "0.3.0"


def get_agent_docs() -> dict[str, _Path]:
    """Get paths to AGENT.md documentation files for AI agent navigation.

    Returns a dict mapping logical names to file paths. Start with 'root'
    for package overview, then drill into specific areas as needed.

    Returns
    -------
    dict[str, Path]
        Mapping of doc names to paths. Keys include:
        - 'root': Package overview and directory map
        - 'features': Feature category index (107 indicators)
        - 'features/{category}': Category-specific signatures
        - 'labeling': ML label generation methods
        - 'bars': Alternative bar sampling

    Example
    -------
    >>> from ml4t.engineer import get_agent_docs
    >>> docs = get_agent_docs()
    >>> print(docs['root'].read_text()[:200])  # Read overview
    """
    pkg_dir = _Path(__file__).parent
    docs = {}

    # Root and package-level
    if (p := pkg_dir / "AGENT.md").exists():
        docs["root"] = p

    # Features index and categories
    features_dir = pkg_dir / "features"
    if (p := features_dir / "AGENT.md").exists():
        docs["features"] = p
    for category_dir in features_dir.iterdir():
        if category_dir.is_dir() and (p := category_dir / "AGENT.md").exists():
            docs[f"features/{category_dir.name}"] = p

    # Other modules
    for module in ["labeling", "bars"]:
        if (p := pkg_dir / module / "AGENT.md").exists():
            docs[module] = p

    return docs

from . import (
    core,
    dataset,
    discovery,
    features,
    labeling,
    outcome,
    pipeline,
    preprocessing,
    relationships,
    store,
    validation,
    visualization,
)
from .api import compute_features
from .dataset import (
    DatasetInfo,
    FoldResult,
    MLDatasetBuilder,
    create_dataset_builder,
)
from .discovery import FeatureCatalog
from .discovery.catalog import features as feature_catalog
from .preprocessing import (
    BaseScaler,
    MinMaxScaler,
    NotFittedError,
    PreprocessingPipeline,
    Preprocessor,
    RobustScaler,
    StandardScaler,
    TransformType,
)

__all__ = [
    # Main API
    "compute_features",
    # Agent navigation
    "get_agent_docs",
    # Feature Discovery (discoverability API)
    "FeatureCatalog",
    "feature_catalog",
    # Dataset builder (leakage-safe train/test preparation)
    "MLDatasetBuilder",
    "create_dataset_builder",
    "FoldResult",
    "DatasetInfo",
    # Preprocessing (leakage-safe scalers)
    "Preprocessor",
    "PreprocessingPipeline",
    "TransformType",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "BaseScaler",
    "NotFittedError",
    # Submodules
    "core",
    "dataset",
    "discovery",
    "features",
    "labeling",
    "outcome",
    "pipeline",
    "preprocessing",
    "relationships",
    "store",
    "validation",
    "visualization",
]
