from __future__ import annotations

from collections.abc import ItemsView, ValuesView
from typing import Generic, Iterator, TypeVar

_VT = TypeVar("_VT")


class ValueTypedDict(dict[str, _VT], Generic[_VT]):
    def __init__(self, **kwargs: _VT):
        super().__init__(**kwargs)

    def __getitem__(self, key: str) -> _VT:
        return super().__getitem__(key)

    def get(self, key: str, default: _VT = None) -> _VT:
        return super().get(key, default)

    def pop(self, key: str) -> _VT:
        return super().pop(key)

    def values(self) -> ValuesView[_VT]:
        return super().values()

    def items(self) -> ItemsView[str, _VT]:
        return super().items()

    def __iter__(self) -> Iterator[str]:
        return super().__iter__()

    def __or__(self, other: dict[str, _VT]) -> ValueTypedDict[_VT]:
        return ValueTypedDict(super().__or__(other))

    def __ior__(self, other: dict[str, _VT]) -> ValueTypedDict[_VT]:
        super().__ior__(other)
        return self
