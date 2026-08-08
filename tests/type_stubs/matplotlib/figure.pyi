from typing import Any

class Figure:
    def __getattr__(self, name: str) -> Any: ...

class SubFigure:
    def __getattr__(self, name: str) -> Any: ...
