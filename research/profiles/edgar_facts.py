"""Pull the EDGAR facts used in the buyer profiles, and print them as YAML-ready rows.

For each public company: latest annual revenue from companyfacts (with the accession and filing
date of the filing it came from) and the most recent 8-K reporting Item 2.01, "Completion of
Acquisition or Disposition of Assets". The profile YAMLs in data/profiles/ cite these rows.

Usage (from the repo root, SEC_USER_AGENT set in .env):
    .venv/bin/python -I research/profiles/edgar_facts.py
"""

from __future__ import annotations

from datetime import date

from acquirer_engine.edgar.client import EdgarClient, filing_dir_url
from acquirer_engine.edgar.parse import (
    filings_as_of,
    iter_filings,
    latest_fact_as_of,
    select_8k_items,
)
from acquirer_engine.env import load_env

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# buyer name in the CSV (or its parent) -> ticker
TICKERS = {
    "HCA Healthcare": "HCA",
    "Ardent Health Services": "ARDT",
    "Amedisys": "AMED",
    "Humana (parent of CenterWell)": "HUM",
    "UnitedHealth Group (parent of Optum)": "UNH",
    "Oracle (parent of Cerner)": "ORCL",
    "Blackstone": "BX",
    "KKR": "KKR",
    "TPG Capital": "TPG",
    # added Oct 23 for the Health IT sample target's Tier 1
    "Evolent Health": "EVH",
    "McKesson": "MCK",
    "Privia Health": "PRVA",
}
AS_OF = date.today()


def main() -> None:
    """Print one block per company: CIK, revenue fact, latest Item 2.01 8-K."""
    load_env()
    with EdgarClient() as client:
        ciks = {row["ticker"]: row["cik_str"] for row in client.company_tickers().values()}
        for name, ticker in TICKERS.items():
            cik = ciks.get(ticker)
            if cik is None:
                print(f"# {name}: ticker {ticker} not found on EDGAR")
                continue
            print(f"# {name} ({ticker}), CIK {cik}")
            revenue = latest_fact_as_of(client.companyfacts(cik), "revenue", AS_OF)
            if revenue is not None:
                print(
                    f"  revenue: {revenue.value / 1e6:,.0f} ($M), FY ending {revenue.end}, "
                    f"form {revenue.form}, accession {revenue.accession}, filed {revenue.filed}"
                )
            filings = filings_as_of(iter_filings(client.submissions(cik), str(cik)), AS_OF)
            for ref in select_8k_items(filings, ["2.01"])[:2]:
                print(
                    f"  8-K item 2.01: accession {ref.accession}, filed {ref.filing_date}, "
                    f"items {','.join(ref.items)}, "
                    f"{filing_dir_url(cik, ref.accession)}/{ref.primary_document}"
                )
        print(f"# EDGAR stats: {client.stats()}")


if __name__ == "__main__":
    main()
