#!/usr/bin/env python3
"""The Chad directory: 25-30K company names from public sources.

Ingests company lists from:
  1. Y Combinator - all batches (public CSV dataset)
  2. Wikipedia - List of unicorn startup companies
  3. Wikipedia - Fortune 1000
  4. Top VC portfolio pages (best-effort link scraping)

Emits directory/companies.jsonl and uploads it to the Azure blob
container so the crawl worker can resolve ATS slugs against it.

Usage:
    python3 scripts/company_directory.py           # build + upload
    python3 scripts/company_directory.py --no-upload
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_PATH = (
    Path(os.environ.get("HO_DIRECTORY_DIR", Path(__file__).resolve().parent / "directory"))
    / "companies.jsonl"
)

YC_CSV_URL = "https://raw.githubusercontent.com/nikshepg/YC-Startup-Directory/main/YC_companies.csv"
WIKI_API = "https://en.wikipedia.org/w/api.php"

VC_PAGES = [
    "https://a16z.com/portfolio/",
    "https://www.sequoiacap.com/our-companies/",
    "https://www.accel.com/companies",
    "https://www.khoslaventures.com/portfolio/",
    "https://www.bvp.com/portfolio",
    "https://www.indexventures.com/companies/",
    "https://www.insightpartners.com/portfolio/",
    "https://www.gv.com/portfolio/",
    "https://www.salesforceventures.com/portfolio/",
    "https://greylock.com/portfolio/",
    "https://www.nea.com/portfolio",
    "https://firstround.com/companies/",
    "https://www.felicis.com/companies",
    "https://500.co/companies",
    "https://www.techstars.com/portfolio/",
    "https://www.coatue.com/portfolio/",
    "https://www.intelcapital.com/portfolio/",
    "https://www.qualcommventures.com/portfolio/",
    "https://visionfund.com/portfolio/",
    "https://www.balderton.com/companies/",
    "https://northzone.com/portfolio/",
    "https://initialized.com/companies/",
    "https://luxcapital.com/companies/",
    "https://www.crv.com/companies/",
    "https://www.foundersfund.com/portfolio/",
    "https://www.sequoia.com/companies/",
    "https://www.andreesenhorowitz.com/companies/",
    "https://www.lightspeedvp.com/companies/",
    "https://www.benchmark.com/companies/",
    "https://www.gener8tor.com/portfolio/",
]

_COMPANY_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_JUNK = {
    "wikipedia",
    "category",
    "file:",
    "image:",
    "template",
    "the",
    "companies",
    "list",
    "unicorn",
    "company",
    "inc",
    "ltd",
    "corporation",
    "group",
    "of",
    "in",
    "and",
    "for",
    "with",
    "founded",
    "valuation",
    "headquarters",
}


def _http_get(url: str, timeout: int = 25) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ho-radar/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().strip(".,")).strip()


def _add(companies: dict[str, dict], name: str, source: str, **extra) -> None:
    name = _clean_name(name)
    if not name or len(name) < 2 or name.lower() in _JUNK:
        return
    key = name.lower()
    entry = companies.setdefault(key, {"name": name, "sources": set()})
    entry["sources"].add(source)
    for k, v in extra.items():
        if v is not None and not entry.get(k):
            entry[k] = v


def fetch_yc(companies: dict[str, dict]) -> None:
    raw = _http_get(YC_CSV_URL)
    for line in raw.splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        name = parts[0].strip().strip('"')
        year = parts[2].strip() if len(parts) > 2 else ""
        _add(companies, name, "yc", year=year)
    print(f"yc: {sum(1 for e in companies.values() if 'yc' in e['sources'])}")


def _wiki_parse(title: str) -> str:
    url = f"{WIKI_API}?action=parse&page={urllib.parse.quote(title)}&prop=wikitext&format=json"
    data = json.loads(_http_get(url))
    return data["parse"]["wikitext"]["*"]


def fetch_unicorns(companies: dict[str, dict]) -> None:
    wt = _wiki_parse("List of unicorn startup companies")
    for m in _COMPANY_LINK_RE.finditer(wt):
        name = m.group(1).strip()
        if "|" in name or "[" in name:
            continue
        _add(companies, name, "unicorn")
    print(f"unicorns: {sum(1 for e in companies.values() if 'unicorn' in e['sources'])}")


def fetch_fortune(companies: dict[str, dict]) -> None:
    for page in (
        "List of largest companies in the United States by revenue",
        "List of largest technology companies by revenue",
    ):
        wt = _wiki_parse(page)
        for m in _COMPANY_LINK_RE.finditer(wt):
            name = m.group(1).strip()
            if "|" in name or "[" in name or ":" in name:
                continue
            _add(companies, name, "fortune")
    print(f"fortune: {sum(1 for e in companies.values() if 'fortune' in e['sources'])}")


def fetch_vc_portfolios(companies: dict[str, dict]) -> None:
    for url in VC_PAGES:
        try:
            html = _http_get(url, timeout=15)
        except Exception:
            continue
        for m in re.finditer(r'<a[^>]*href="[^"]*"[^>]*>([^<]{2,60})</a>', html):
            name = _clean_name(m.group(1))
            if not name or re.search(r"[|{}\[\]=]", name):
                continue
            _add(companies, name, "vc")
    print(f"vc: {sum(1 for e in companies.values() if 'vc' in e['sources'])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    companies: dict[str, dict] = {}
    fetch_yc(companies)
    fetch_unicorns(companies)
    fetch_fortune(companies)
    fetch_vc_portfolios(companies)

    OUT_PATH.parent.mkdir(exist_ok=True)
    rows = []
    for entry in companies.values():
        entry["sources"] = sorted(entry["sources"])
        rows.append(entry)
    rows.sort(key=lambda r: r["name"].lower())
    with OUT_PATH.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"TOTAL: {len(rows)} companies → {OUT_PATH}")

    if args.no_upload:
        return

    env: dict[str, str] = {}
    for key in ("AZURE_STORAGE_ACCOUNT", "AZURE_STORAGE_KEY", "AZURE_CONTAINER"):
        if os.environ.get(key):
            env[key] = os.environ[key]
    if "AZURE_STORAGE_ACCOUNT" not in env:
        env_file = PROJECT / "scripts" / ".watchdog.env"
        if not env_file.exists():
            print("WARN: azure creds missing (env or scripts/.watchdog.env), skipping upload")
            return
        for line in env_file.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                env.setdefault(k.strip(), v.strip())
    try:
        from azure.storage.blob import BlobServiceClient

        cs = (
            "DefaultEndpointsProtocol=https;"
            f"AccountName={env['AZURE_STORAGE_ACCOUNT']};"
            f"AccountKey={env['AZURE_STORAGE_KEY']};"
            "EndpointSuffix=core.windows.net"
        )
        cc = BlobServiceClient.from_connection_string(cs).get_container_client(
            env.get("AZURE_CONTAINER", "radar-index")
        )
        cc.get_blob_client("directory/companies.jsonl").upload_blob(
            OUT_PATH.read_bytes(), overwrite=True
        )
        print("uploaded → directory/companies.jsonl")
    except Exception as exc:
        print(f"WARN: upload failed: {exc}")


if __name__ == "__main__":
    main()
