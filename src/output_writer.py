"""Output writer: generate jobs.md table."""

from datetime import UTC, datetime


def compute_days_ago(date_str: str | None) -> str:
    if not date_str:
        return "?"
    try:
        dt = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        delta = datetime.now(UTC) - dt
        days = delta.days
        if days == 0:
            return "Today"
        if days == 1:
            return "1d ago"
        return f"{days}d ago"
    except ValueError, TypeError:
        return "?"


def write_md(jobs: list[dict], output_path: str = "jobs.md") -> None:
    header = (
        "| # | Role | Company | JD Match | Shortlist% | Salary | Posted | Location | Apply |\n"
        "|---|------|---------|----------|------------|--------|--------|----------|-------|"
    )

    rows = []
    for i, j in enumerate(jobs, 1):
        role = j.get("role", "?")
        company = j.get("company", "?")
        match_pct = f"{j.get('match_percent', '?')}%"
        shortlist = f"{j.get('shortlist_probability', '?')}%"
        salary = str(j.get("salary") or "-")
        posted = compute_days_ago(j.get("posted_date"))
        location = j.get("location", "?")
        link = j.get("apply_link") or j.get("source_url", "")
        link_md = f"[Apply]({link})" if link else "-"

        row = (
            f"| {i} | {role} | {company} | {match_pct} | {shortlist} | "
            f"{salary} | {posted} | {location} | {link_md} |"
        )
        rows.append(row)

    with open(output_path, "w") as f:
        f.write(f"# Job Matches\n\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n\n")
        f.write(header + "\n")
        f.write("\n".join(rows) + "\n")
        f.write(f"\n*{len(jobs)} positions matched*\n")

    print(f"\n  Wrote {len(jobs)} jobs to {output_path}")
