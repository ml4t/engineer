from typing import Any

class Axes:
    figure: Any
    def __getattr__(self, name: str) -> Any: ...
