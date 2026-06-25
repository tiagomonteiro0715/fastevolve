import math
import re
from collections import Counter


class CodeEmbedder:
    def __init__(self):
        self.index: list[tuple[Counter, int]] = []

    def embed(self, program) -> Counter:
        return Counter(re.findall(r"\w+", program.code))

    def add(self, vector: Counter, program_id: int):
        self.index.append((vector, program_id))

    @staticmethod
    def _cos(a: Counter, b: Counter) -> float:
        num = sum(a[t] * b[t] for t in set(a) & set(b))
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return num / (na * nb) if na and nb else 0.0

    def nearest(self, vector: Counter, *, k: int) -> list[int]:
        scored = sorted(((self._cos(vector, v), pid) for v, pid in self.index), reverse=True)
        return [pid for _, pid in scored[:k]]
