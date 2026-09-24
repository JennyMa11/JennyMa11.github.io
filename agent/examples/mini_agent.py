"""A deterministic, dependency-free example of an Agent tool loop.

Run: python3 agent/examples/mini_agent.py
Replace DemoPolicy.next with a model adapter to experiment with an LLM.
"""

from dataclasses import dataclass


DOCUMENTS = {
    "react": "ReAct alternates reasoning, acting, and observations.",
    "rag": "RAG retrieves external evidence for generation.",
}


@dataclass(frozen=True)
class Decision:
    kind: str
    name: str = ""
    query: str = ""
    answer: str = ""


class DemoPolicy:
    """A predictable stand-in for the model, so this example runs offline."""

    def next(self, goal, history):
        if not history:
            term = "react" if "ReAct" in goal else "rag"
            return Decision(kind="tool", name="search", query=term)
        observation = history[-1]
        if observation["status"] == "ok":
            return Decision(kind="final", answer=observation["text"])
        return Decision(kind="final", answer="没有找到可验证证据。")


def run(goal, policy=None, max_steps=4):
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    policy = policy or DemoPolicy()
    history = []
    for step in range(max_steps):
        decision = policy.next(goal, history)
        if decision.kind == "final":
            return {"answer": decision.answer, "steps": step + 1, "trace": history}
        if decision.kind != "tool" or decision.name != "search":
            raise ValueError("unknown action")
        if not isinstance(decision.query, str) or not decision.query.strip():
            raise ValueError("invalid query")
        found = DOCUMENTS.get(decision.query.lower())
        observation = {
            "call_id": f"call-{step + 1}",
            "tool": "search",
            "query": decision.query,
            "status": "ok" if found else "not_found",
            "text": found or "",
        }
        history.append(observation)
    return {"answer": "已达到步骤上限，任务交由人工检查。", "steps": max_steps, "trace": history}


if __name__ == "__main__":
    import json

    print(json.dumps(run("解释 ReAct"), ensure_ascii=False, indent=2))
