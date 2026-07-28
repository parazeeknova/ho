"""Rich CLI: spinners, progress, panels, live status for the pipeline."""

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()


class PipelineDisplay:
    def __init__(self) -> None:
        self.phase = ""
        self.subtitle = ""
        self.metrics: dict[str, str] = {}
        self.log_lines: list[str] = []

    def set_phase(self, phase: str, subtitle: str = "") -> None:
        self.phase = phase
        self.subtitle = subtitle

    def update_metrics(self, **kwargs: str) -> None:
        self.metrics.update(kwargs)

    def log(self, msg: str) -> None:
        self.log_lines.append(msg)
        if len(self.log_lines) > 20:
            self.log_lines.pop(0)

    def render(self) -> Panel:
        content = Text()

        if self.phase:
            content.append(f"\n[bold cyan]{self.phase}[/bold cyan]")
            if self.subtitle:
                content.append(f"\n[dim]{self.subtitle}[/dim]")
            content.append("\n")

        if self.metrics:
            content.append("\n")
            for k, v in self.metrics.items():
                content.append(f"  [bold]{k}:[/bold] {v}\n")

        if self.log_lines:
            content.append("\n[dim]")
            for line in self.log_lines[-8:]:
                content.append(f"  {line}\n")
            content.append("[/dim]")

        return Panel(
            content,
            title="[bold]ho[/bold] — job matching pipeline",
            border_style="cyan",
            box=box.ROUNDED,
        )


def spinner_task(desc: str, fn, *args, **kwargs):
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]{desc}...", total=None)
        result = fn(*args, **kwargs)
        progress.remove_task(task)
        return result


def show_resume(text: str, sections: list[str]) -> None:
    console.print(
        Panel(
            Text(text[:1500] + ("..." if len(text) > 1500 else ""), style="dim"),
            title=(
                f"[bold green]Resume[/] — {len(text)} chars, "
                f"{len(sections)} sections: {', '.join(sections)}"
            ),
            border_style="green",
            box=box.ROUNDED,
        )
    )


def show_results(jobs: list[dict]) -> None:
    if not jobs:
        console.print("[yellow]No matching jobs found.[/yellow]")
        return

    table = Table(
        title=f"[bold]Top {len(jobs)} Job Matches[/bold]",
        box=box.SIMPLE_HEAVY,
        border_style="cyan",
    )
    table.add_column("#", style="dim", width=3)
    table.add_column("Role", style="bold cyan")
    table.add_column("Company", style="green")
    table.add_column("Match", justify="right")
    table.add_column("Shortlist", justify="right")
    table.add_column("Location")
    table.add_column("Apply")

    for i, j in enumerate(jobs, 1):
        role = str(j.get("role", "?"))[:40]
        company = str(j.get("company", "?"))[:25]
        match_pct = f"{j.get('match_percent', '?')}%"
        shortlist = f"{j.get('shortlist_probability', '?')}%"
        location = str(j.get("location", "?"))[:20]
        link = j.get("apply_link") or j.get("source_url", "")
        apply_cell = f"[link={link}]apply[/link]" if link else "-"

        color = "green" if j.get("match_percent", 0) >= 70 else "yellow"
        table.add_row(
            str(i),
            role,
            company,
            f"[{color}]{match_pct}[/{color}]",
            f"[{color}]{shortlist}[/{color}]",
            location,
            apply_cell,
        )

    console.print(table)
    console.print(f"\n[dim]Wrote {len(jobs)} positions to jobs.md[/dim]")
