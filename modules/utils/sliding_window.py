from collections import deque
from typing import TypeVar, Generic, Hashable

T = TypeVar("T", bound=Hashable)

class SlidingWindow(Generic[T]):
    def __init__(self, window_size: int, initial_val: T, update_threshold: float = 0.7):
        self.__window: deque[T] = deque()
        self.__counts: dict[T, int] = dict()

        self.__window_size = window_size
        self.__max_val = initial_val
        self.__update_threshold = update_threshold
    
    def append(self, val: T):
        self.__window.append(val)
        
        if val not in self.__counts:
            self.__counts[val] = 0
        self.__counts[val] += 1
        
        if len(self.__window) <= self.__window_size:
            return
        
        left_val = self.__window.popleft()
        self.__counts[left_val] -= 1
        if self.__counts[left_val] == 0:
            del self.__counts[left_val]

        max_val = max(self.__counts, key=self.__counts.get)
        if self.__counts[max_val] / self.__window_size >= self.__update_threshold:
            self.__max_val = max_val

    def get_max_val(self) -> T:
        return self.__max_val
