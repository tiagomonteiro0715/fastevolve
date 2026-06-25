import heapq
import random
from collections import deque
from .program import Program
from .embedder import CodeEmbedder


class ProgramDatabase:
    def __init__(self, config):
        self.config = config
        self.islands: list[dict] = [dict() for _ in range(config.num_islands)]
        self.by_id: dict[int, Program] = {}
        self.heap: list[tuple[float, int]] = []
        self.migrations = [deque(maxlen=config.migration_size) for _ in range(config.num_islands)]
        self.embedder = CodeEmbedder()
        self.step = 0

    def _cell(self, behavior):
        if not behavior:
            return (0,)
        return tuple(min(self.config.cell_bins - 1, int(b * self.config.cell_bins)) for b in behavior)

    def seed(self, code: str, *, island: int = 0) -> Program:
        p = Program(code=code, island=island)
        self.by_id[p.id] = p
        self.islands[island][("seed", p.id)] = p
        heapq.heappush(self.heap, (0.0, p.id))
        self.embedder.add(self.embedder.embed(p), p.id)
        return p

    def add(self, child_program: Program, results):
        child_program.fitness = results.fitness
        child_program.behavior = results.behavior
        cell = self._cell(results.behavior)
        grid = self.islands[child_program.island % self.config.num_islands]
        if cell not in grid or grid[cell].fitness < child_program.fitness:
            grid[cell] = child_program
        self.by_id[child_program.id] = child_program
        heapq.heappush(self.heap, (-child_program.fitness, child_program.id))
        self.embedder.add(self.embedder.embed(child_program), child_program.id)
        self.step += 1
        if self.step % self.config.migration_every == 0:
            self._migrate()

    def _migrate(self):
        n = self.config.num_islands
        for i, grid in enumerate(self.islands):
            if grid:
                self.migrations[(i + 1) % n].append(max(grid.values(), key=lambda p: p.fitness))

    def sample(self):
        top = heapq.nsmallest(self.config.top_k, self.heap)
        live = [(f, pid) for f, pid in top if pid in self.by_id] or [(0.0, next(iter(self.by_id)))]
        parent = self.by_id[random.choice(live)[1]]
        ids = self.embedder.nearest(self.embedder.embed(parent), k=self.config.num_inspirations + 1)
        insp = [self.by_id[i] for i in ids if i != parent.id and i in self.by_id][: self.config.num_inspirations]
        return parent, insp
