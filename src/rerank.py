"""
rerank.py (ponytail) -- dùng sentence-transformers.CrossEncoder, không tự viết
tokenize/batch/no_grad tay như bản trước.
"""


def load_reranker(model_name: str = "AITeamVN/Vietnamese_Reranker", device: str = "cpu", **kw):
    """
    Cửa duy nhất để load reranker. Tự nhận diện họ model -> notebook chỉ cần đổi
    RERANKER_MODEL, không phải đổi dòng import.
    """
    if _is_qwen(model_name):
        from rerank_qwen import load_qwen_reranker

        return load_qwen_reranker(model_name, device=device, **kw)

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device, max_length=kw.pop("max_length", 1024), **kw)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"[load_reranker] {model_name}: {n_params:,} ({n_params / 1e9:.3f}B)")
    if n_params > 3_000_000_000:
        raise ValueError(f"{model_name} vượt ngân sách 3B ({n_params / 1e9:.2f}B)")
    return model


def _is_qwen(name: str) -> bool:
    return "qwen" in name.lower()


def rerank_question(question: str, candidate_ids: list, corpus: dict, model, k: int = 5) -> list[str]:
    """model: bất kỳ object nào có .predict(list[[q, doc]]) -> list[score] (CrossEncoder thật hoặc fake để test)."""
    valid_ids = [str(c) for c in candidate_ids if str(c) in corpus]
    if not valid_ids:
        return []
    pairs = [[question, corpus[cid]["passage"]] for cid in valid_ids]
    scores = model.predict(pairs)
    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [cid for cid, _ in ranked[:k]]


def rerank_all(questions: dict, candidates_dict: dict, corpus: dict, model, k: int = 5) -> dict:
    return {
        str(qid): rerank_question(q, candidates_dict.get(str(qid), []), corpus, model, k)
        for qid, q in questions.items()
    }


if __name__ == "__main__":
    fake_corpus = {
        "1": {"passage": "văn bản về đăng ký xe máy"},
        "2": {"passage": "văn bản về nghĩa vụ quân sự"},
        "3": {"passage": "văn bản không liên quan gì cả"},
    }

    class FakeModel:  # thay CrossEncoder thật, test logic sort/cut, không cần internet
        def predict(self, pairs):
            return [sum(1 for w in q.split() if w in d) for q, d in pairs]

    m = FakeModel()
    r1 = rerank_question("đăng ký xe máy cần gì", ["3", "1", "2"], fake_corpus, m, k=2)
    assert r1[0] == "1", r1

    r2 = rerank_question("đăng ký xe máy", ["999", "1"], fake_corpus, m, k=5)
    assert r2 == ["1"], r2

    r3 = rerank_all(
        {"q1": "đăng ký xe máy", "q2": "nghĩa vụ quân sự"},
        {"q1": ["1", "2", "3"], "q2": ["1", "2", "3"]},
        fake_corpus, m, k=2,
    )
    assert r3["q1"][0] == "1" and r3["q2"][0] == "2", r3

    print("OK -", r1, r2, r3)

    assert _is_qwen("Qwen/Qwen3-Reranker-0.6B") and _is_qwen("infgrad/Prism-Qwen3.5-Reranker-2B")
    assert not _is_qwen("AITeamVN/Vietnamese_Reranker") and not _is_qwen("BAAI/bge-reranker-v2-m3")
    print("Test _is_qwen OK - dispatch đúng họ model")
