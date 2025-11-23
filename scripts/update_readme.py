#!/usr/bin/env python3
"""
Update README progress table and generate badge.svg
Run from repo root.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHALLENGES_DIR = REPO_ROOT / "challanges"
README = REPO_ROOT / "README.md"
BADGE_SVG = REPO_ROOT / "badge.svg"
TOTAL_DAYS = 100

# ----------------------
# Helpers
# ----------------------

def count_days():
    if not CHALLENGES_DIR.exists():
        return 0, []
    entries = sorted([
        d for d in CHALLENGES_DIR.iterdir()
        if d.is_dir() and re.match(r'^\d+_', d.name)
    ])
    return len(entries), entries


def make_folder_link(d):
    folder = d.name
    path = f"challanges/{folder}/"
    return f"[`{folder}`]({path})"


def extract_existing_topic_and_desc():
    """
    Parse existing README progress table and return two dicts:
      topics = { "1": "Vector Addition", ... }
      desc   = { "1": "Basic CUDA kernel ...", ... }
    """
    text = README.read_text(encoding='utf-8')

    # Capture: Day | (ignored Folder column) | Topic | Short description |
    # Use non-greedy matches for safety; work line-by-line.
    pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*.*?\|\s*(.*?)\s*\|\s*(.*?)\s*\|",
        re.M
    )

    topics = {}
    desc = {}
    for day, topic, description in pattern.findall(text):
        topic_clean = topic.strip()
        desc_clean = description.strip()
        if topic_clean not in ("...", ""):
            topics[day] = topic_clean
        if desc_clean not in ("...", ""):
            desc[day] = desc_clean
    return topics, desc


def generate_progress_block(n_done, entries):
    pct = int((n_done / TOTAL_DAYS) * 100)

    # Load any user-provided topics/descriptions to preserve them
    existing_topics, existing_desc = extract_existing_topic_and_desc()

    # No leading newline here (prevents extra blank line before block)
    header = (
        "<!-- PROGRESS_TABLE_START -->\n"
        "| Day | Folder | Topic | Short description |\n"
        "|-----|--------|-------|-------------------|\n"
    )

    rows = []
    for d in entries:
        m = re.match(r'^(\d+)_+(.*)', d.name)
        day = m.group(1)
        # Default topic derived from folder name, but preserved if README already has one
        default_title = m.group(2).replace('_', ' ') if m else d.name
        topic = existing_topics.get(day, default_title)

        folder_link = make_folder_link(d)
        short_desc = existing_desc.get(day, "")

        rows.append(f"| {day} | {folder_link} | {topic} | {short_desc} |")

    if n_done < TOTAL_DAYS:
        rows.append("| ... | ... | ... | ... |")

    # Exactly one newline before Progress line and no trailing newline after END (prevents extra blank lines)
    footer = (
        "\n\nProgress: **Day {} / {} ({}%)**\n"
        "<!-- PROGRESS_TABLE_END -->"
    ).format(n_done, TOTAL_DAYS, pct)

    return header + "\n".join(rows) + footer


def update_readme(n_done, entries):
    text = README.read_text(encoding='utf-8')
    start_marker = "<!-- PROGRESS_TABLE_START -->"
    end_marker = "<!-- PROGRESS_TABLE_END -->"

    new_block = generate_progress_block(n_done, entries)

    if start_marker in text and end_marker in text:
        # Replace the whole existing block (from START to END) with the new one
        new_text = re.sub(
            f"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
            new_block,
            text,
            flags=re.S
        )
    else:
        # Append with exactly two newlines before the block for readability
        new_text = text.rstrip() + "\n\n" + new_block

    README.write_text(new_text, encoding='utf-8')
    print("README updated")


def gen_badge_svg(n_done):
    pct = int((n_done / TOTAL_DAYS) * 100)
    left = f"Day {n_done}/{TOTAL_DAYS}"
    right = f"{pct}%"

    left_w = 8 * len(left) + 30
    right_w = 8 * len(right) + 20
    total_w = left_w + right_w

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_w}" height="20">
  <linearGradient id="g" x2="0" y2="100%">
    <stop offset="0" stop-color="#bbb" stop-opacity=".1"/>
    <stop offset="1" stop-opacity=".1"/>
  </linearGradient>
  <rect rx="3" width="{total_w}" height="20" fill="#555"/>
  <rect rx="3" x="{left_w}" width="{right_w}" height="20" fill="#4c1"/>
  <path fill="#4c1" d="M{left_w} 0h4v20h-4z"/>
  <rect rx="3" width="{total_w}" height="20" fill="url(#g)"/>
  <g fill="#fff" text-anchor="middle"
     font-family="DejaVu Sans,Verdana,Arial" font-size="11">
    <text x="{left_w/2}" y="14">{left}</text>
    <text x="{left_w + right_w/2}" y="14">{right}</text>
  </g>
</svg>"""

    BADGE_SVG.write_text(svg, encoding='utf-8')
    print("badge.svg written")


def main():
    n_done, entries = count_days()
    update_readme(n_done, entries)
    gen_badge_svg(n_done)


if __name__ == "__main__":
    main()