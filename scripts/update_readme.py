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


def extract_existing_descriptions():
    """
    Parse existing README progress table and return:
    { "1": "my desc", "2": "..." }
    """
    text = README.read_text(encoding='utf-8')

    pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*.+?\|\s*.+?\|\s*(.*?)\s*\|"
    )

    desc = {}
    for day, description in pattern.findall(text):
        if description.strip() not in ("...", ""):
            desc[day] = description.strip()
    return desc


def generate_progress_block(n_done, entries):
    pct = int((n_done / TOTAL_DAYS) * 100)

    # Load any user-provided descriptions
    existing_desc = extract_existing_descriptions()

    header = (
        "\n<!-- PROGRESS_TABLE_START -->\n"
        "| Day | Folder | Topic | Short description |\n"
        "|-----|--------|-------|-------------------|\n"
    )

    rows = []
    for d in entries:
        m = re.match(r'^(\d+)_+(.*)', d.name)
        day = m.group(1)
        title = m.group(2).replace('_', ' ') if m else d.name

        folder_link = make_folder_link(d)
        short_desc = existing_desc.get(day, "")

        rows.append(f"| {day} | {folder_link} | {title} | {short_desc} |")

    if n_done < TOTAL_DAYS:
        rows.append("| ... | ... | ... | ... |")

    # NOTE: Add blank line before footer
    footer = (
        "\n\nProgress: **Day {} / {} ({}%)**\n"
        "<!-- PROGRESS_TABLE_END -->\n"
    ).format(n_done, TOTAL_DAYS, pct)

    return header + "\n".join(rows) + footer


def update_readme(n_done, entries):
    text = README.read_text(encoding='utf-8')
    start_marker = "<!-- PROGRESS_TABLE_START -->"
    end_marker = "<!-- PROGRESS_TABLE_END -->"

    new_block = generate_progress_block(n_done, entries)

    if start_marker in text and end_marker in text:
        new_text = re.sub(
            f"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
            new_block,
            text,
            flags=re.S
        )
    else:
        new_text = text + "\n\n" + new_block

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
