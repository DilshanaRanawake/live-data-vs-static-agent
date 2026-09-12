from static_agent import static_answer
from full_agent import full_agent_answer
from questions import QUESTIONS
import json
import time
import os

RESULTS_FILE = "comparison_results.json"

# resume support: load existing results if the file already exists
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)
    done_questions = {r["question"] for r in results}
else:
    results = []
    done_questions = set()

for i, q in enumerate(QUESTIONS, 1):
    if q in done_questions:
        print(f"[{i}/{len(QUESTIONS)}] already done, skipping: {q}")
        continue

    print(f"[{i}/{len(QUESTIONS)}] {q}")

    start = time.time()
    static = static_answer(q)
    static_time = time.time() - start
    print(f"  static done in {static_time:.1f}s")

    start = time.time()
    live = full_agent_answer(q)
    live_time = time.time() - start
    print(f"  live-search done in {live_time:.1f}s")

    results.append({
        "question": q,
        "static_only": static,
        "static_time_seconds": round(static_time, 1),
        "with_live_search": live,
        "live_time_seconds": round(live_time, 1),
    })

    # save after every question, not just at the end
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

print("\nDone. Review comparison_results.json and score each answer against ground_truth.json.")