from typing import Any


class ModelRegistry:
    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def register(self, name: str, model: Any) -> None:
        self._models[name] = model

    def get(self, name: str) -> Any:
        if name not in self._models:
            raise ValueError(f"Model bulunamadı: {name}")
        return self._models[name]

    def has_model(self, name: str) -> bool:
        return name in self._models