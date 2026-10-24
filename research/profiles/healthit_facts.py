"""Pull EDGAR evidence for the Health IT sample target's public Tier 1 buyers (Evolent, McKesson,
Privia).

For each company: every 8-K since 2025 with its items (1.01 agreement, 2.01 completed deal, 5.01
change in control: a pending sale changes the buyer's status), and sentences from the latest 10-K
on acquisition strategy and footprint. The profile YAMLs cite the accession numbers printed here.

Usage (from the repo root, SEC_USER_AGENT set in .env):
    .venv/bin/python -I research/profiles/healthit_facts.py
"""

from __future__ import annotations

import re
from datetime import date

from acquirer_engine.edgar.client import EdgarClient, filing_document_url
from acquirer_engine.edgar.parse import html_to_text, iter_filings
from acquirer_engine.env import load_env

TICKERS = {"Evolent Health": "EVH", "McKesson": "MCK", "Privia Health": "PRVA"}
SINCE = date(2025, 1, 1)
KEYWORDS = re.compile(
    r"(acquisition strategy|acquire|tuck-in|bolt-on|strategic alternatives|merger agreement|"
    r"\d+ states|operate in \d+|markets across)",
    re.I,
)


def main() -> None:
    load_env()
    with EdgarClient() as client:
        ciks = {row["ticker"]: row["cik_str"] for row in client.company_tickers().values()}
        for name, ticker in TICKERS.items():
            cik = ciks[ticker]
            filings = list(iter_filings(client.submissions(cik)))
            print(f"# {name} ({ticker}), CIK {cik}")
            for f in filings:
                if f.form == "8-K" and f.filing_date >= SINCE:
                    print(
                        f"  8-K {f.filing_date} items {','.join(f.items)} accession {f.accession}"
                    )
            tenk = next(f for f in filings if f.form == "10-K")
            url = filing_document_url(cik, tenk.accession, tenk.primary_document)
            text = html_to_text(client.get_text(url))
            print(f"  10-K {tenk.filing_date} accession {tenk.accession} {url}")
            hits = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", text)
                if KEYWORDS.search(s) and 40 < len(s) < 400
            ]
            for s in hits[:12]:
                print(f"    - {s}")


if __name__ == "__main__":
    main()
