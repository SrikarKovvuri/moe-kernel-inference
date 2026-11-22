"""Prepare a small deterministic public-text probe set; raw text stays under runs/."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "d5678bc8ac0cdd79dbd5e44d4130271018bcec4e"
URL = (
    f"https://raw.githubusercontent.com/pytorch/examples/{REVISION}/"
    "word_language_model/data/wikitext-2/test.txt"
)


def select_probes(payload, min_characters=500, join_paragraphs=False):
    if min_characters < 1:
        raise ValueError("min_characters must be positive")
    lines = payload.decode("utf-8").splitlines()
    candidates = []
    for index, line in enumerate(lines, 1):
        text, end = line.strip(), index
        minimum_start = 500 if join_paragraphs else min_characters
        if len(text) < minimum_start or text.startswith("="):
            continue
        if join_paragraphs:
            while len(text) < min_characters and end < len(lines):
                text += "\n" + lines[end]
                end += 1
        if len(text) >= min_characters:
            candidates.append((index, end, text))
    if len(candidates) < 3:
        raise ValueError("expected at least three long WikiText test paragraphs")
    selected = [candidates[i] for i in (0, len(candidates) // 2, len(candidates) - 1)]
    return [
        dict(
            name=f"wikitext-line-{line}",
            text=text,
            source="WikiText-2 test, tokenized text; three deterministic contiguous probes",
            source_url=URL,
            source_revision=REVISION,
            source_line=line,
            source_end_line=end,
            source_selection_min_characters=min_characters,
            source_join_paragraphs=join_paragraphs,
            source_sha256=hashlib.sha256(payload).hexdigest(),
            source_license="CC BY-SA 3.0 / GFDL (WikiText dataset card)",
        )
        for line, end, text in selected
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/wikitext-probes.json"))
    p.add_argument("--min-characters", type=int, default=500)
    p.add_argument("--join-paragraphs", action="store_true")
    args = p.parse_args()
    with urlopen(URL, timeout=60) as response:
        payload = response.read(5 * 1024 * 1024 + 1)
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError("unexpectedly large test split")
    probes = select_probes(payload, args.min_characters, args.join_paragraphs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(probes, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(probes)} probes; source SHA256={hashlib.sha256(payload).hexdigest()}")


if __name__ == "__main__":
    main()
