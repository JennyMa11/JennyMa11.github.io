"""Behavior checks for the CPU scheduling example."""

import unittest

from mini_serving import BlockPool, MiniEngine, Request


class MiniServingTests(unittest.TestCase):
    def test_chunked_prefill_and_continuous_batching(self):
        engine = MiniEngine(total_blocks=8, block_size=4, token_budget=4)
        engine.add(Request("long", list(range(7)), 3))
        engine.add(Request("short", list(range(3)), 2))

        first = engine.step()
        self.assertEqual([(name, n) for name, n, _ in first], [("long", 4)])
        self.assertEqual(engine.running[0].output, [])  # no intermediate sampling

        while engine.waiting or engine.running:
            self.assertTrue(engine.step())
        self.assertEqual(
            {request.request_id: len(request.output) for request in engine.finished},
            {"long": 3, "short": 2},
        )
        self.assertEqual(len(engine.pool.free), engine.pool.total_blocks)

    def test_cancel_releases_blocks(self):
        engine = MiniEngine(total_blocks=2, block_size=4, token_budget=4)
        engine.add(Request("cancel_me", list(range(6)), 2))
        engine.step()
        self.assertEqual(len(engine.pool.free), 1)
        self.assertTrue(engine.cancel("cancel_me"))
        self.assertFalse(engine.cancel("cancel_me"))
        self.assertEqual(len(engine.pool.free), 2)

    def test_oom_does_not_partial_allocate(self):
        pool = BlockPool(total_blocks=1, block_size=4)
        self.assertEqual(pool.reserve("one", 4), [0])
        self.assertIsNone(pool.reserve("one", 8))
        self.assertEqual(pool.tables["one"], [0])
        pool.release("one")
        self.assertEqual(len(pool.free), 1)


if __name__ == "__main__":
    unittest.main()
