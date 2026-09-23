"""CPU-only teaching model of token scheduling and paged KV ownership.

Run: python3 examples/mini_serving.py
This simulates model outputs; it is not an LLM inference implementation.
"""

from collections import deque
from dataclasses import dataclass, field


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    computed: int = 0
    output: list[int] = field(default_factory=list)
    status: str = "waiting"

    def remaining_prompt(self):
        return max(0, len(self.prompt) - self.computed)


class BlockPool:
    def __init__(self, total_blocks, block_size):
        self.total_blocks = total_blocks
        self.block_size = block_size
        self.free = deque(range(total_blocks))
        self.tables = {}

    def reserve(self, request_id, total_tokens):
        table = self.tables.get(request_id, [])
        required = (total_tokens + self.block_size - 1) // self.block_size
        missing = required - len(table)
        if missing > len(self.free):
            return None
        self.tables.setdefault(request_id, table)
        for _ in range(missing):
            table.append(self.free.popleft())
        self.check()
        return list(table)

    def release(self, request_id):
        for block in self.tables.pop(request_id, []):
            self.free.append(block)
        self.check()

    def check(self):
        owned = [block for table in self.tables.values() for block in table]
        all_blocks = list(self.free) + owned
        assert len(all_blocks) == self.total_blocks
        assert len(set(all_blocks)) == self.total_blocks


class MiniEngine:
    def __init__(self, total_blocks=8, block_size=4, token_budget=4):
        self.pool = BlockPool(total_blocks, block_size)
        self.token_budget = token_budget
        self.waiting = deque()
        self.running = []
        self.finished = []

    def add(self, request):
        assert request.prompt and request.max_new_tokens > 0
        self.waiting.append(request)

    def cancel(self, request_id):
        for queue in (self.waiting, self.running):
            for request in list(queue):
                if request.request_id == request_id:
                    queue.remove(request)
                    request.status = "cancelled"
                    self.pool.release(request_id)
                    self.finished.append(request)
                    return True
        return False

    def _admit(self, request, budget):
        remaining = request.remaining_prompt()
        want = min(remaining if remaining else 1, budget)
        if not want:
            return None
        table = self.pool.reserve(request.request_id, request.computed + want)
        if table is None:
            return None
        return request, want, table

    def step(self):
        budget = self.token_budget
        plan = []
        for request in list(self.running):
            item = self._admit(request, budget)
            if item:
                plan.append(item)
                budget -= item[1]
        for request in list(self.waiting):
            item = self._admit(request, budget)
            if item:
                self.waiting.remove(request)
                self.running.append(request)
                request.status = "running"
                plan.append(item)
                budget -= item[1]
            if budget == 0:
                break

        for request, count, table in plan:
            request.computed += count
            # Intermediate prefill chunks only write KV; no sampling.
            if request.remaining_prompt():
                continue
            # A deterministic stand-in for model forward + sampler.
            request.output.append(1000 + len(request.output))
            if len(request.output) == request.max_new_tokens:
                request.status = "finished"
                self.running.remove(request)
                self.pool.release(request.request_id)
                self.finished.append(request)
        return [(r.request_id, n, table) for r, n, table in plan]


def demo():
    engine = MiniEngine(total_blocks=8, block_size=4, token_budget=4)
    engine.add(Request("A", list(range(7)), 3))
    engine.add(Request("B", list(range(3)), 2))
    tick = 0
    while engine.waiting or engine.running:
        plan = engine.step()
        if not plan:
            raise RuntimeError("No progress: KV capacity or token budget exhausted")
        print(f"step={tick} plan={plan} free_blocks={len(engine.pool.free)}")
        tick += 1
    assert {request.request_id: len(request.output) for request in engine.finished} == {"A": 3, "B": 2}
    assert len(engine.pool.free) == engine.pool.total_blocks


if __name__ == "__main__":
    demo()
