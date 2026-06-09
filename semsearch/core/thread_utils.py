import threading
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeSet(Generic[T]):
    """A thread-safe set wrapper."""

    def __init__(self) -> None:
        self._set: set[T] = set()
        self._lock = threading.Lock()

    def add(self, item: T) -> None:
        with self._lock:
            self._set.add(item)

    def __contains__(self, item: object) -> bool:
        with self._lock:
            return item in self._set

    def add_if_absent(self, item: T) -> bool:
        """Atomically add item if not already present. Returns True if added."""
        with self._lock:
            if item in self._set:
                return False
            self._set.add(item)
            return True

    def __len__(self) -> int:
        with self._lock:
            return len(self._set)


class ThreadSafeDict(Generic[K, V]):
    """A thread-safe dict wrapper."""

    def __init__(self) -> None:
        self._dict: dict[K, V] = {}
        self._lock = threading.Lock()

    def get_or_insert(self, key: K, default_factory: Callable[[], V]) -> V:
        """Atomically return the value for key, inserting default_factory() if absent."""
        with self._lock:
            if key not in self._dict:
                self._dict[key] = default_factory()
            return self._dict[key]

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._dict

    def __len__(self) -> int:
        with self._lock:
            return len(self._dict)
