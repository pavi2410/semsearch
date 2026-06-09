import threading
from typing import Generic, TypeVar

T = TypeVar("T")


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
