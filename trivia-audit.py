#!/usr/bin/env python3
"""
trivia-audit.py — Shorten genuinely bloated questions in all question JSON files.
Target: under 20 words. Brief context clauses are fine and encouraged — don't over-trim.
Logs every change made.
"""

import json
import os
import anthropic
from pathlib import Path

QUESTIONS_DIR = Path(__file__).parent / "content" / "questions"
MAX_WORDS = 20

SKIP = {"schema.json", "QUESTION_RULES.md"}

client = anthropic.Anthropic()

def word_count(text: str) -> int:
    return len(text.split())

def shorten_question(question: str) -> str:
    prompt = (
        "Shorten this trivia question to under 20 words. "
        "Cut Wikipedia-style preambles and excessive scene-setting. "
        "But keep a short context clause if it aids learning — don't over-trim. "
        "A question like 'Which cat breed, named after the Isle of Man, is born without a tail?' is perfect. "
        "Reply with ONLY the shortened question, no explanation.\n\n"
        f"Question: {question}"
    )
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip().strip('"')

def audit_file(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)

    if "questions" not in data:
        return []

    changes = []
    for q in data["questions"]:
        original = q["question"]
        if word_count(original) > MAX_WORDS:
            shortened = shorten_question(original)
            q["question"] = shortened
            changes.append({
                "id": q.get("id", "?"),
                "file": path.name,
                "before": original,
                "after": shortened,
                "words_before": word_count(original),
                "words_after": word_count(shortened),
            })
            print(f"  [{q.get('id')}] {word_count(original)}w → {word_count(shortened)}w")
            print(f"    BEFORE: {original}")
            print(f"    AFTER:  {shortened}")

    if changes:
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    return changes

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set")
        raise SystemExit(1)

    files = sorted([
        p for p in QUESTIONS_DIR.glob("*.json")
        if p.name not in SKIP and not p.name.endswith("_raw.json")
    ])

    print(f"Auditing {len(files)} files (target: <{MAX_WORDS} words)...\n")

    all_changes = []
    for path in files:
        with open(path) as f:
            data = json.load(f)
        questions = data.get("questions", [])
        long_qs = [q for q in questions if word_count(q.get("question", "")) > MAX_WORDS]
        if not long_qs:
            print(f"✓ {path.name}")
            continue
        print(f"\n→ {path.name} ({len(long_qs)} to shorten)")
        changes = audit_file(path)
        all_changes.extend(changes)

    print(f"\n{'='*60}")
    print(f"Done. {len(all_changes)} questions shortened across {len(set(c['file'] for c in all_changes))} files.")

if __name__ == "__main__":
    main()
