"""Offline exact-token retrieval, permission filtering and evidence metrics.

Run: python3 agent/examples/retrieval_lab.py
This deliberately simple overlap scorer is neither BM25 nor semantic/vector search.
"""

import json
import re

DOCS = [
    {"id": "D1", "version": "v2", "scope": "public", "text": "standard shipping threshold 100 fee 10"},
    {"id": "D2", "version": "v2", "scope": "public", "text": "member shipping fee 0"},
    {"id": "D3", "version": "v1", "scope": "archive", "text": "standard shipping threshold 200 fee 10"},
    {"id": "D4", "version": "v2", "scope": "internal", "text": "member shipping internal account exception"},
]
QUESTIONS = [
    {"query": "standard shipping threshold", "gold": {"D1"}},
    {"query": "member shipping fee", "gold": {"D2"}},
    {"query": "standard member shipping", "gold": {"D1", "D2"}},
    {"query": "refund deadline", "gold": set()},
]


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query, k=2):
    if k < 1:
        raise ValueError("k must be positive")
    query_terms = tokens(query)
    visible = [doc for doc in DOCS if doc["scope"] == "public" and doc["version"] == "v2"]
    scored = [(len(query_terms & tokens(doc["text"])), doc) for doc in visible]
    scored.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
    return [doc for score, doc in scored[:k] if score > 0]


def evaluate(k):
    reports = []
    for question in QUESTIONS:
        hits = retrieve(question["query"], k)
        found = {doc["id"] for doc in hits}
        gold = question["gold"]
        reports.append({"query": question["query"], "hits": [doc["id"] for doc in hits],
                        "recall": len(found & gold) / len(gold) if gold else None,
                        "should_abstain": not gold, "retrieved_nothing": not hits})
    recalls = [r["recall"] for r in reports if r["recall"] is not None]
    return {"k": k, "mean_recall_answerable": sum(recalls) / len(recalls), "questions": reports}


if __name__ == "__main__":
    print(json.dumps([evaluate(1), evaluate(2)], ensure_ascii=False, indent=2))
