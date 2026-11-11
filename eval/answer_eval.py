"""
Answer eval: run every golden question through the real chain, then have the judge
grade correctness (vs the reference) and faithfulness (vs the retrieved context).

Run from the repo root:
    python -m eval.answer_eval

Each result is appended to RESULTS_PATH as soon as it is done. Re-running skips
questions already in the file, so a rate-limit crash can just be resumed.
Delete the file to start over (e.g. after changing the prompt or a model).
"""

import argparse
import json
import os
import re

from settings import ANSWER_MODEL, INDEX_PATH, JUDGE_MODEL
from src import chainer, init_embeddings, init_llm, load_vectorstore

GOLDEN_PATH = "eval/golden_set.jsonl"
RESULTS_PATH = "eval/results.jsonl"
REFUSAL = "the notes don't cover this"

CORRECTNESS_PROMPT = """You are grading an answer from a study assistant against a reference answer.

Question: {question}

Reference answer: {reference}

Answer to grade: {answer}

The answer is correct if it contains the key facts of the reference and nothing that contradicts it.
Extra correct detail is fine. A missing key fact counts as incorrect.
Ignore page citations like [p. 3] and formatting.
Reply with JSON only: {{"verdict": "yes" or "no", "reason": "<one sentence>"}}"""

FAITHFULNESS_PROMPT = """You are checking whether an answer is supported by the context it was given.

Context:
{context}

Answer: {answer}

The answer is faithful if every factual claim in it is stated in, or directly follows from, the context.
A claim that is not in the context makes the answer unfaithful, even if the claim is true.
Ignore page citations like [p. 3] and formatting.
Reply with JSON only: {{"verdict": "yes" or "no", "reason": "<one sentence>"}}"""


def is_refusal(answer):
    """True if the answer is the prompt's fixed refusal reply."""
    return answer.replace("’", "'").strip().strip('"').lower().startswith(REFUSAL)


def judge_call(judge, prompt):
    """Ask the judge and parse its JSON verdict. On a bad reply, verdict is None and raw is kept."""
    raw = judge.invoke(prompt).content
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        parsed = json.loads(match.group(0))
        verdict = str(parsed["verdict"]).strip().lower()
        if verdict in ("yes", "no"):
            return {"verdict": verdict, "reason": parsed.get("reason", "")}
    except (AttributeError, KeyError, ValueError):
        pass
    return {"verdict": None, "reason": "", "raw": raw}


def main():
    parser = argparse.ArgumentParser(description="Answer and judge the golden set.")
    parser.add_argument("--out", default=RESULTS_PATH, help="results file (JSONL)")
    args = parser.parse_args()

    with open(GOLDEN_PATH, encoding="utf-8") as f:
        golden = [json.loads(line) for line in f if line.strip()]
    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f if line.strip()}
    todo = [q for q in golden if q["id"] not in done]
    print(f"{len(done)} already done, {len(todo)} to go. Answer: {ANSWER_MODEL}, judge: {JUDGE_MODEL}")

    chain = chainer(load_vectorstore(INDEX_PATH, init_embeddings()), init_llm(ANSWER_MODEL))
    judge = init_llm(JUDGE_MODEL)

    for q in todo:
        response = chain.invoke({"input": q["question"]})
        answer = response["answer"]
        docs = response["context"]
        refused = is_refusal(answer)

        correct = faithful = None
        if not refused:
            if q["pages"]:  # correctness only means something for answerable questions
                correct = judge_call(judge, CORRECTNESS_PROMPT.format(
                    question=q["question"], reference=q["answer"], answer=answer))
            context = "\n\n".join(f"[p. {d.metadata['page']}]\n{d.page_content}" for d in docs)
            faithful = judge_call(judge, FAITHFULNESS_PROMPT.format(context=context, answer=answer))

        result = {
            "id": q["id"], "type": q["type"], "question": q["question"],
            "reference": q["answer"], "evidence": q["pages"],
            "answer": answer, "refused": refused,
            "retrieved": [d.metadata["page"] for d in docs],
            "correct": correct, "faithful": faithful,
            "answer_model": ANSWER_MODEL, "judge_model": JUDGE_MODEL,
        }
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        print(q["id"], "refused" if refused else
              f"correct={correct and correct['verdict']} faithful={faithful['verdict']}")


if __name__ == "__main__":
    main()
