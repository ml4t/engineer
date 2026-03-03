# Feature Discovery

ML4T Engineer provides two complementary discovery APIs: the **Feature Registry** for programmatic metadata access, and the **Feature Catalog** for interactive exploration with filtering and search.

## Feature Registry

The registry is the metadata backbone — every feature registers its name, category, parameters, input requirements, and validation status.

```python
from ml4t.engineer.core.registry import get_registry

registry = get_registry()
```

### List Features

```python
# All 120 features (sorted alphabetically)
all_features = registry.list_all()

# By category
momentum = registry.list_by_category("momentum")  # 31 features

# By property
normalized = registry.list_normalized()            # 37 ML-ready features
ta_lib = registry.list_ta_lib_compatible()         # 59 validated features
```

### Inspect Metadata

```python
meta = registry.get("rsi")

meta.name           # "rsi"
meta.category       # "momentum"
meta.description    # "Relative Strength Index"
meta.formula        # "RSI = 100 - 100/(1 + RS), RS = AvgGain/AvgLoss"
meta.parameters     # {"period": 14}
meta.input_type     # "OHLCV"
meta.output_type    # "indicator"
meta.normalized     # True
meta.value_range    # (0, 100)
meta.ta_lib_compatible  # True
meta.dependencies   # []
meta.references     # ["Wilder, 1978"]
meta.tags           # ["oscillator", "overbought", "oversold"]
```

### Get Dependencies

Some features depend on others (e.g., `stochrsi` depends on `rsi`):

```python
deps = registry.get_dependencies("stochrsi")  # ["rsi"]
```

`compute_features` resolves these automatically via topological sort.

## Feature Catalog

The catalog wraps the registry with higher-level filtering and full-text search:

```python
from ml4t.engineer import feature_catalog
```

### Filtered Listing

```python
# Single filter
feature_catalog.list(category="volatility")

# Multiple filters (AND logic)
feature_catalog.list(
    category="momentum",
    normalized=True,
    ta_lib_compatible=True,
)

# All available filters
feature_catalog.list(
    category=None,              # str: filter by category
    normalized=None,            # bool: ML-ready features only
    ta_lib_compatible=None,     # bool: TA-Lib validated only
    tags=None,                  # list[str]: features matching any tag
    input_type=None,            # str: "OHLCV", "close", "returns", etc.
    output_type=None,           # str: "indicator", "bands", etc.
    has_dependencies=None,      # bool: features with/without dependencies
    limit=None,                 # int: max results
)
```

### Full-Text Search

Search across feature names, descriptions, tags, and formulas:

```python
results = feature_catalog.search("volatility estimator")
# [("parkinson_volatility", 0.65), ("garman_klass_volatility", 0.45), ("rogers_satchell_volatility", 0.45), ...]

results = feature_catalog.search("trend strength")
# [("trend_intensity_index", 0.92), ("adx", 0.78), ...]

results = feature_catalog.search("spread")
# [("realized_spread", 1.30), ("roll_spread_estimator", 1.30)]
```

Returns a list of `(feature_name, relevance_score)` tuples, sorted by relevance.

### Describe

Get a dict summary of any feature:

```python
info = feature_catalog.describe("yang_zhang_volatility")
# {
#     "name": "yang_zhang_volatility",
#     "category": "volatility",
#     "description": "Yang-Zhang Volatility - combines overnight and intraday volatility",
#     "parameters": {},
#     "normalized": False,
#     "ta_lib_compatible": False,
#     "input_type": "close",
#     "value_range": None,
#     "dependencies": [],
#     "tags": [],
# }
```

### Browse Categories and Tags

```python
feature_catalog.categories()
# ['cross_asset', 'math', 'microstructure', 'ml', 'momentum',
#  'price_transform', 'regime', 'risk', 'statistics', 'trend',
#  'volatility', 'volume']

feature_catalog.tags()
# ['efficient', 'illiquidity', 'ma', 'microstructure', 'normalized',
#  'ohlc', 'oscillator', 'overbought', 'oversold', 'spread', ...]
```

## Metadata Fields Reference

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Unique feature identifier |
| `category` | `str` | Feature category |
| `description` | `str` | One-line description |
| `formula` | `str` | Mathematical formula |
| `parameters` | `dict[str, Any]` | Default parameters |
| `input_type` | `str` | Required input columns (`"OHLCV"`, `"close"`, etc.) |
| `output_type` | `str` | Output type (`"indicator"`, `"bands"`, etc.) |
| `normalized` | `bool \| None` | Whether output is bounded |
| `value_range` | `tuple[float, float] \| None` | Output range if normalized |
| `ta_lib_compatible` | `bool` | Validated against TA-Lib at 1e-6 |
| `dependencies` | `list[str]` | Other features this depends on |
| `references` | `list[str]` | Academic references |
| `tags` | `list[str]` | Searchable tags |
| `lookback` | `Callable` | Function returning minimum lookback period |

> **Book**: *ML for Trading, 3rd ed.* — Ch7 `10_ml4t_library_ecosystem.py` explores the registry metadata for RSI, ATR, and Garman-Klass, then demonstrates `feature_catalog.search()` and filtered listing.
