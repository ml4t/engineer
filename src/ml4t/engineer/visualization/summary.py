"""Plot export utilities.

For feature analysis visualizations (IC, importance, drift),
use ``ml4t.diagnostic.visualization``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def export_plot(
    fig: Figure,
    output_path: str | Path,
    dpi: int = 300,
    bbox_inches: str = "tight",
    **kwargs: Any,
) -> None:
    """Export matplotlib figure to file.

    Saves figure to PNG or PDF with configurable quality settings.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    output_path : str | Path
        Output file path. Format determined by extension (.png, .pdf, etc.).
    dpi : int, default 300
        Resolution in dots per inch. Higher values = better quality but larger files.
        - 150: Draft quality
        - 300: Publication quality (default)
        - 600: High-resolution print
    bbox_inches : str, default "tight"
        Bounding box specification. "tight" removes extra whitespace.
    **kwargs
        Additional keyword arguments passed to fig.savefig().

    Raises
    ------
    ValueError
        If output_path has unsupported extension.
    OSError
        If file cannot be written (permissions, disk space, etc.).

    Examples
    --------
    >>> from ml4t.engineer.visualization import export_plot
    >>>
    >>> # Create and export plot
    >>> fig = plt.figure()
    >>> # ... create plot ...
    >>> export_plot(fig, "analysis.png", dpi=300)
    >>>
    >>> # High-quality PDF
    >>> export_plot(fig, "analysis.pdf", dpi=600)
    """
    output_path = Path(output_path)

    # Validate extension
    valid_extensions = {".png", ".pdf", ".svg", ".jpg", ".jpeg", ".eps", ".ps"}
    if output_path.suffix.lower() not in valid_extensions:
        msg = f"Unsupported format: {output_path.suffix}. Use one of {valid_extensions}"
        raise ValueError(msg)

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save figure
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    except OSError as e:
        msg = f"Failed to save figure to {output_path}: {e}"
        raise OSError(msg) from e
