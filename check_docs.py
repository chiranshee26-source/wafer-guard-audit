"""Check that every headline number in README.md and AUDIT.md matches results.json.

Run after executing the notebook:  python check_docs.py
"""
import json
import sys

r = json.load(open("results.json"))
keys = ["f1", "precision", "recall", "roc_auc", "pr_auc", "test_prevalence", "threshold",
        "f1_ci_low", "f1_ci_high", "pr_auc_ci_low", "pr_auc_ci_high",
        "baseline_f1", "baseline_pr_auc", "baseline_roc_auc"]
expected = {k: f"{r[k]:.3f}" for k in keys}
expected["confusion"] = f"{r['TN']} / {r['FP']} / {r['FN']} / {r['TP']}"

ok = True
for doc in ("README.md", "AUDIT.md"):
    text = open(doc, encoding="utf-8").read()
    for k, v in expected.items():
        if v not in text:
            ok = False
            print(f"{doc}: expected {k} = {v} but it does not appear")
print("all numbers match results.json" if ok else "\nUpdate the documents with the numbers above.")
sys.exit(0 if ok else 1)
