"""
Score the saved answer eval. No LLM calls; rerun as often as you like.

Run from the repo root:
    python -m eval.score               # metrics (+ judge agreement once the audit is labelled)
    python -m eval.score --sample 30   # write the judge audit file for hand-labelling

Audit: fill each row's "human" with "yes" or "no" without looking at the judge,
then run the plain command again. The judge's verdict is not in the audit file.
"""

import argparse
import json
import os
import random
import re

from settings import OUTPUT_PATH

RESULTS_PATH = "eval/results.jsonl"
AUDIT_PATH = "eval/judge_audit.json"
KINDS = ("correct", "faithful")


def verdict(judged):
    """'yes', 'no', or None. Recovers the verdict from replies whose JSON didn't parse."""
    if judged is None:
        return None
    if judged["verdict"]:
        return judged["verdict"]
    match = re.search(r'"verdict"\s*:\s*"?(yes|no)', judged.get("raw", ""), re.I)
    return match.group(1).lower() if match else None


def cited_pages(answer):
    """Page numbers inside [p. N] / [p. 3, p. 5] citations."""
    return [int(n) for group in re.findall(r"\[([^\]]*)\]", answer) for n in re.findall(r"p\.\s*(\d+)", group)]


def page_text(page):
    with open(os.path.join(OUTPUT_PATH, "output_text", f"page_{page:03d}.txt"), encoding="utf-8") as f:
        return f.read()


def pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def judged(group, kind):
    """Share of 'yes' among the rows in group that have a parsed verdict for kind."""
    vs = [v for v in (verdict(r[kind]) for r in group) if v]
    return pct(vs.count("yes"), len(vs)) if vs else "-"


def retrieval(group):
    """Hit rate, MRR and recall over the answerable rows in group, as strings ('-' if none)."""
    group = [r for r in group if r["evidence"]]
    if not group:
        return "-", "-", "-"
    ranks = [next((i + 1 for i, p in enumerate(r["retrieved"]) if p in r["evidence"]), None) for r in group]
    mrr = sum(1 / x for x in ranks if x) / len(group)
    recall = sum(len(set(r["evidence"]) & set(r["retrieved"])) / len(set(r["evidence"])) for r in group) / len(group)
    return pct(sum(x is not None for x in ranks), len(group)), f"{mrr:.2f}", f"{recall:.2f}"


def write_audit(rows, n):
    if os.path.exists(AUDIT_PATH):
        print(f"{AUDIT_PATH} already exists; delete it first if you really want a new sample.")
        return
    items = [(r, k) for r in rows for k in KINDS if verdict(r[k])]
    rng = random.Random(0)
    no = [x for x in items if verdict(x[0][x[1]]) == "no"]
    yes = [x for x in items if verdict(x[0][x[1]]) == "yes"]
    rng.shuffle(no)
    rng.shuffle(yes)
    picked = no[:n // 2]
    picked += yes[:n - len(picked)]
    rng.shuffle(picked)
    audit = []
    for r, kind in picked:
        row = {"id": r["id"], "kind": kind, "question": r["question"], "answer": r["answer"]}
        if kind == "correct":
            row["reference"] = r["reference"]
        else:  # what the judge saw: the retrieved pages
            row["context"] = {p: page_text(p) for p in dict.fromkeys(r["retrieved"])}
        row["human"] = ""
        audit.append(row)
    with open(AUDIT_PATH, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(picked)} items to {AUDIT_PATH} ({len(no[:n // 2])} judged no). Fill in 'human'.")


def report_audit(rows):
    if not os.path.exists(AUDIT_PATH):
        return
    with open(AUDIT_PATH, encoding="utf-8") as f:
        audit = json.load(f)
    by_id = {r["id"]: r for r in rows}
    pairs = [(verdict(by_id[a["id"]][a["kind"]]), a["human"].strip().lower(), a) for a in audit
             if a["human"].strip().lower() in ("yes", "no")]
    print(f"\nJudge audit: {len(pairs)}/{len(audit)} labelled")
    if not pairs:
        return
    n = len(pairs)
    agree = sum(j == h for j, h, _ in pairs)
    p_judge = sum(j == "yes" for j, _, _ in pairs) / n
    p_human = sum(h == "yes" for _, h, _ in pairs) / n
    chance = p_judge * p_human + (1 - p_judge) * (1 - p_human)
    kappa = (agree / n - chance) / (1 - chance) if chance < 1 else float("nan")
    print(f"  agreement {pct(agree, n)}, Cohen's kappa {kappa:.2f}")
    for j, h, a in pairs:
        if j != h:
            print(f"  disagree: {a['id']} {a['kind']} judge={j} you={h}")


def main():
    parser = argparse.ArgumentParser(description="Score eval/results.jsonl.")
    parser.add_argument("--results", default=RESULTS_PATH)
    parser.add_argument("--sample", type=int, help="write N judge verdicts to the audit file")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if args.sample:
        write_audit(rows, args.sample)
        return

    answerable = [r for r in rows if r["evidence"]]
    unanswerable = [r for r in rows if not r["evidence"]]
    answered = [r for r in rows if not r["refused"]]
    for r in rows:
        r["hit"] = bool(set(r["evidence"]) & set(r["retrieved"]))
    wrong = [r for r in answerable if r["refused"]]
    missed = [r for r in unanswerable if not r["refused"]]
    cites = [(r, p) for r in answered for p in cited_pages(r["answer"])]
    ans_cites = [(r, p) for r, p in cites if r["evidence"]]

    print(f"{len(rows)} questions ({len(answerable)} answerable, {len(unanswerable)} unanswerable)")
    print(f"answer model: {rows[0]['answer_model']}   judge: {rows[0]['judge_model']}")

    # Headline numbers
    k = max(len(r["retrieved"]) for r in rows)
    hit, mrr, recall = retrieval(rows)
    print("\nSummary")
    for name, value, note in [
        (f"hit@{k}", hit, "answerable questions with an evidence page retrieved"),
        ("MRR", mrr, "1 / rank of the first evidence page, averaged"),
        (f"recall@{k}", recall, "share of each question's evidence pages retrieved"),
        ("correct", judged(answered, "correct"), "answers that match the reference"),
        ("faithful", judged(answered, "faithful"), "answers with every claim in the context"),
        ("wrong refusals", pct(len(wrong), len(answerable)), "answerable questions it refused"),
        ("missed refusals", pct(len(missed), len(unanswerable)), "unanswerable questions it answered"),
        ("cited", pct(sum(bool(cited_pages(r["answer"])) for r in answered), len(answered)), "answers with a [p. N]"),
        ("cited retrieved", pct(sum(p in r["retrieved"] for r, p in cites), len(cites)), "citations to a page it was given"),
        ("cited evidence", pct(sum(p in r["evidence"] for r, p in ans_cites), len(ans_cites)), "citations to a golden evidence page"),
    ]:
        print(f"  {name:<17}{value:>14}   {note}")
    unparsed = sum(r[k] is not None and verdict(r[k]) is None for r in rows for k in KINDS)
    if unparsed:
        print(f"  ({unparsed} judge replies could not be parsed; excluded)")

    # Same, split by question type
    print(f"\nBy type\n  {'type':<11}{'n':>3}{'hit':>14}{'MRR':>6}{'recall':>8}{'refused':>9}{'correct':>15}{'faithful':>15}")
    for t in sorted({r["type"] for r in rows}):
        group = [r for r in rows if r["type"] == t]
        hit, mrr, recall = retrieval(group)
        print(f"  {t:<11}{len(group):>3}{hit:>14}{mrr:>6}{recall:>8}{sum(r['refused'] for r in group):>9}"
              f"{judged(group, 'correct'):>15}{judged(group, 'faithful'):>15}")

    # Which questions failed, and where
    buckets = {
        "wrong, retrieval miss": [r for r in answerable if not r["refused"] and not r["hit"] and verdict(r["correct"]) == "no"],
        "wrong but faithful, retrieval hit": [r for r in answerable if not r["refused"] and r["hit"]
                                              and verdict(r["correct"]) == "no" and verdict(r["faithful"]) == "yes"],
        "unfaithful": [r for r in answered if verdict(r["faithful"]) == "no"],
        "correct, no evidence page (check labels)": [r for r in answerable if not r["refused"] and not r["hit"]
                                                     and verdict(r["correct"]) == "yes"],
        "refused, retrieval hit": [r for r in wrong if r["hit"]],
        "refused, retrieval miss": [r for r in wrong if not r["hit"]],
        "missed refusal": missed,
    }
    print("\nDiagnosis")
    for name, group in buckets.items():
        if group:
            print(f"  {name:<42}{', '.join(r['id'] for r in group)}")

    report_audit(rows)


if __name__ == "__main__":
    main()
