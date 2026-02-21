#!/usr/bin/env python3
"""
trivia-gen.py — Generate trivia question batches using the Anthropic Python SDK.

Requires ANTHROPIC_API_KEY environment variable.

Usage:
    python3 trivia-gen.py                          # run all categories in CATEGORIES list
    python3 trivia-gen.py --categories animals art  # specific slugs only
    python3 trivia-gen.py --reset                   # clear checkpoint and start over
    python3 trivia-gen.py --list                    # show available slugs and exit
"""

import os
import sys
import json
import time
import logging
import re
import argparse
import anthropic
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "content" / "questions"
CHECKPOINT  = BASE_DIR / "content" / "trivia-gen-checkpoint.json"
LOG_FILE    = Path("/var/log/trivia-gen.log")
REF_FILE    = OUTPUT_DIR / "true-crime.json"

# ── Config ─────────────────────────────────────────────────────────────────

SLEEP_BATCH     = 300   # 5 minutes between batches within a category
SLEEP_CATEGORY  = 1200  # 20 minutes between categories
MODEL       = "claude-sonnet-4-6"

# ── Category registry ──────────────────────────────────────────────────────
# (slug, display_name, id_prefix)

CATEGORIES = [
    ("animals",       "Animals",        "ani"),
    ("art",           "Art",            "art"),
    ("biology",       "Biology",        "bio"),
    ("chemistry",     "Chemistry",      "chem"),
    ("food",          "Food",           "food"),
    ("geography",     "Geography",      "geo"),
    ("human-body",    "The Human Body", "hb"),
    ("inventions",    "Inventions",     "inv"),
    ("languages",     "Languages",      "lang"),
    ("literature",    "Literature",     "lit"),
    ("mathematics",   "Mathematics",    "math"),
    ("movies",        "Movies",         "mov"),
    ("music",         "Music",          "mus"),
    ("mythology",     "Mythology",      "myth"),
    ("physics",       "Physics",        "phys"),
    ("space",         "Space",          "spc"),
    ("sports",        "Sports",         "spt"),
    ("technology",    "Technology",     "tech"),
    ("world-records", "World Records",  "wr"),
]

CATEGORY_MAP = {slug: (slug, name, prefix) for slug, name, prefix in CATEGORIES}

# ── Reference examples (loaded once from true-crime.json) ──────────────────

def load_reference_examples() -> str:
    if not REF_FILE.exists():
        return ""
    with open(REF_FILE) as f:
        data = json.load(f)
    lines = []
    for q in data["questions"][:8]:
        correct = next(a["text"] for a in q["answers"] if a.get("correct"))
        wrong   = [a["text"] for a in q["answers"] if not a.get("correct")]
        lines.append(
            f"{q['id']} | {q['question']} | "
            f"correct: {correct} | "
            f"wrong: {' / '.join(wrong[:3])}"
        )
    return "\n".join(lines)

REFERENCE_EXAMPLES = None  # loaded lazily

# ── Prompt builder ─────────────────────────────────────────────────────────

def build_prompt(category_name: str, id_prefix: str, batch_num: int,
                 reference: str, prior_questions: list[dict] | None = None) -> str:
    start = 1 + (batch_num - 1) * 25
    end   = start + 24

    if batch_num == 1:
        distribution = (
            f"  • {id_prefix}_001 – {id_prefix}_010 : EASY (10 questions)\n"
            f"  • {id_prefix}_011 – {id_prefix}_020 : MEDIUM (10 questions)\n"
            f"  • {id_prefix}_021 – {id_prefix}_025 : HARD (5 questions)"
        )
    else:
        distribution = (
            f"  • {id_prefix}_026 – {id_prefix}_035 : EASY (10 questions)\n"
            f"  • {id_prefix}_036 – {id_prefix}_045 : MEDIUM (10 questions)\n"
            f"  • {id_prefix}_046 – {id_prefix}_050 : HARD (5 questions)"
        )

    ids = [f"{id_prefix}_{str(i).zfill(3)}" for i in range(start, end + 1)]

    already_covered = ""
    if batch_num == 2 and prior_questions:
        lines = [f"  {q['id']}: {q['question']} ({q['correct']})" for q in prior_questions]
        already_covered = (
            "\nALREADY COVERED IN BATCH 1 — do not repeat these topics or correct answers:\n"
            + "\n".join(lines)
            + "\n"
        )

    return f"""Generate exactly 25 trivia questions for the party game category: {category_name}

DIFFICULTY DISTRIBUTION FOR THIS BATCH:
{distribution}

IDs to use (in order): {ids[0]} through {ids[-1]}

OUTPUT FORMAT — one question per line, exactly this structure:
[id] | [question] | correct: [answer] | wrong: [ans1] / [ans2] / [ans3]

{already_covered}RULES:
- Output ONLY the 25 question lines — no headers, no commentary, no blank lines
- 4 answers per question: 1 correct, 3 plausible wrong answers
- Wrong answers must be genuinely plausible — not obviously incorrect
- Questions should be accurate and factual
- Mix question styles: who/what/when/where/which
- Do not repeat topics across the batch

EXAMPLE OUTPUT (match this quality and format exactly):
{reference}

Generate the {category_name} questions now:"""


# ── Compact-format parser ──────────────────────────────────────────────────

LINE_RE = re.compile(
    r'^(?P<id>[\w_]+)\s*\|\s*'
    r'(?P<question>.+?)\s*\|\s*'
    r'correct:\s*(?P<correct>.+?)\s*\|\s*'
    r'wrong:\s*(?P<wrong>.+?)\s*$',
    re.IGNORECASE
)


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line:
        return None
    m = LINE_RE.match(line)
    if not m:
        return None
    wrong_parts = [w.strip() for w in m.group("wrong").split("/") if w.strip()]
    if len(wrong_parts) != 3:
        return None
    return {
        "id":       m.group("id").strip(),
        "question": m.group("question").strip(),
        "correct":  m.group("correct").strip(),
        "wrong":    wrong_parts,
    }


def parse_response(text: str) -> tuple[list[dict], list[str]]:
    questions, errors = [], []
    for line in text.strip().splitlines():
        result = parse_line(line)
        if result:
            questions.append(result)
        elif line.strip():
            errors.append(line[:120])
    return questions, errors


# ── Validator ──────────────────────────────────────────────────────────────

def validate_batch(questions: list[dict], expected: int = 25) -> list[str]:
    issues = []
    if len(questions) != expected:
        issues.append(f"Expected {expected} questions, got {len(questions)}")
    seen_ids = set()
    for q in questions:
        if q["id"] in seen_ids:
            issues.append(f"Duplicate ID: {q['id']}")
        seen_ids.add(q["id"])
        for field in ("question", "correct"):
            if not q.get(field, "").strip():
                issues.append(f"{q['id']}: empty {field}")
        if len(q.get("wrong", [])) != 3:
            issues.append(f"{q['id']}: expected 3 wrong answers")
    return issues


# ── JSON builder ───────────────────────────────────────────────────────────

def infer_difficulty(q_id: str) -> str:
    m = re.search(r'(\d+)$', q_id)
    if not m:
        return "medium"
    n = int(m.group(1))
    if n <= 10 or 26 <= n <= 35:
        return "easy"
    if n <= 20 or 36 <= n <= 45:
        return "medium"
    return "hard"


def questions_to_json(questions: list[dict], category_name: str) -> dict:
    qs = []
    for q in questions:
        qs.append({
            "id": q["id"],
            "question": q["question"],
            "answers": [
                {"text": q["correct"],   "correct": True},
                {"text": q["wrong"][0],  "correct": False},
                {"text": q["wrong"][1],  "correct": False},
                {"text": q["wrong"][2],  "correct": False},
            ],
            "difficulty": infer_difficulty(q["id"]),
            "category": category_name,
        })
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for q in qs:
        counts[q["difficulty"]] += 1
    return {
        "category": category_name,
        "metadata": {
            "totalQuestions": len(qs),
            "difficulty": counts,
            "version": "1.0",
            "created": datetime.now().strftime("%Y-%m-%d"),
            "updated": datetime.now().strftime("%Y-%m-%d"),
        },
        "questions": qs,
    }


# ── Claude CLI call ────────────────────────────────────────────────────────

_client: anthropic.Anthropic | None = None

def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client

def call_claude(prompt: str, log: logging.Logger) -> str:
    """Call the Anthropic API and return the text result."""
    log.debug(f"Calling API ({MODEL})…")
    message = get_client().messages.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ── Checkpoint helpers ─────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {"completed": [], "partial": {}}


def save_checkpoint(cp: dict) -> None:
    CHECKPOINT.write_text(json.dumps(cp, indent=2))


def mark_batch_done(cp: dict, slug: str, batch_num: int,
                    questions: list[dict]) -> None:
    cp.setdefault("partial", {}).setdefault(slug, {})[str(batch_num)] = questions
    save_checkpoint(cp)


def mark_category_done(cp: dict, slug: str) -> None:
    if slug not in cp["completed"]:
        cp["completed"].append(slug)
    cp.get("partial", {}).pop(slug, None)
    save_checkpoint(cp)


# ── Logging setup ──────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    log = logging.getLogger("trivia-gen")
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    try:
        fh = logging.FileHandler(LOG_FILE)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    except PermissionError:
        log.warning(f"Cannot write to {LOG_FILE} — logging to console only")
    return log


# ── Category generation ────────────────────────────────────────────────────

def generate_category(slug: str, name: str, prefix: str,
                      cp: dict, log: logging.Logger) -> None:
    global REFERENCE_EXAMPLES
    if REFERENCE_EXAMPLES is None:
        REFERENCE_EXAMPLES = load_reference_examples()

    out_path = OUTPUT_DIR / f"{slug}_raw.json"
    log.info(f"[{name}] Starting → {out_path.name}")

    # Recover partial work from checkpoint
    partial = cp.get("partial", {}).get(slug, {})
    all_questions: list[dict] = []
    if "1" in partial:
        log.info(f"[{name}] Resuming — batch 1 recovered from checkpoint "
                 f"({len(partial['1'])} questions)")
        all_questions.extend(partial["1"])

    for batch_num in (1, 2):
        if str(batch_num) in partial:
            log.info(f"[{name}] Batch {batch_num} already in checkpoint, skipping")
            continue

        log.info(f"[{name}] Generating batch {batch_num}/2…")
        prior = all_questions if batch_num == 2 else None
        prompt = build_prompt(name, prefix, batch_num, REFERENCE_EXAMPLES, prior)

        try:
            raw_text = call_claude(prompt, log)
        except Exception as e:
            log.error(f"[{name}] Batch {batch_num} API error: {e}")
            raise

        questions, parse_errors = parse_response(raw_text)

        if parse_errors:
            log.warning(f"[{name}] Batch {batch_num}: "
                        f"{len(parse_errors)} unparseable line(s):")
            for err in parse_errors[:5]:
                log.warning(f"    !! {err}")

        issues = validate_batch(questions)
        if issues:
            log.warning(f"[{name}] Batch {batch_num} validation issues:")
            for issue in issues:
                log.warning(f"    !! {issue}")
            if len(questions) < 20:
                raise ValueError(
                    f"[{name}] Batch {batch_num} only produced {len(questions)} "
                    f"questions — aborting category"
                )

        log.info(f"[{name}] Batch {batch_num}: {len(questions)} questions OK")
        all_questions.extend(questions)
        mark_batch_done(cp, slug, batch_num, questions)

        if batch_num == 1:
            log.info(f"[{name}] Sleeping {SLEEP_BATCH}s before batch 2…")
            time.sleep(SLEEP_BATCH)

    output = questions_to_json(all_questions, name)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    log.info(f"[{name}] Written {len(all_questions)} questions → {out_path}")
    mark_category_done(cp, slug)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate trivia questions via the Claude CLI (Max subscription)"
    )
    parser.add_argument(
        "--categories", nargs="+", metavar="SLUG",
        help="Only generate these slugs (default: all not yet completed)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear checkpoint and regenerate everything"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print available category slugs and exit"
    )
    args = parser.parse_args()

    if args.list:
        for slug, name, prefix in CATEGORIES:
            print(f"{slug:<20}  {name}")
        return

    log = setup_logging()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        log.error("ANTHROPIC_API_KEY is not set")
        sys.exit(1)

    if args.reset and CHECKPOINT.exists():
        CHECKPOINT.unlink()
        log.info("Checkpoint cleared")

    cp = load_checkpoint()
    already_done = cp.get("completed", [])

    if args.categories:
        targets = []
        for slug in args.categories:
            if slug not in CATEGORY_MAP:
                log.warning(f"Unknown slug '{slug}' — skipping")
                continue
            targets.append(CATEGORY_MAP[slug])
    else:
        targets = [(s, n, p) for s, n, p in CATEGORIES if s not in already_done]

    if not targets:
        log.info("Nothing to do — all categories complete. Use --reset to regenerate.")
        return

    log.info(f"Categories to generate ({len(targets)}): "
             f"{', '.join(n for _, n, _ in targets)}")

    failed = []
    for i, (slug, name, prefix) in enumerate(targets):
        if slug in cp.get("completed", []):
            log.info(f"[{name}] Already complete, skipping")
            continue
        try:
            generate_category(slug, name, prefix, cp, log)
        except Exception as e:
            log.error(f"[{name}] Failed: {e}")
            failed.append(slug)
            continue
        if i < len(targets) - 1:
            log.info(f"Sleeping {SLEEP_CATEGORY}s before next category…")
            time.sleep(SLEEP_CATEGORY)

    if failed:
        log.warning(f"Failed categories: {failed}")
        sys.exit(1)
    else:
        log.info("All categories completed successfully.")


if __name__ == "__main__":
    main()
