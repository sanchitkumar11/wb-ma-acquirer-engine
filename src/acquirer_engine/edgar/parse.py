"""Parsers for EDGAR JSON payloads and filing documents.

Everything here is a pure function over already-fetched bytes, so it is fully
testable from recorded fixtures with no network involved.

The as-of filters live here too, right next to the parsing, because that is where
a leak would be introduced: it is easy to write a loop over ``filings.recent``
and forget that the list runs up to today. :func:`filings_as_of` makes the filter
the obvious path rather than an extra step a caller has to remember.
"""

from __future__ import annotations

import html
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date

#: 8-K item reporting completion of an acquisition *or disposition* of assets.
#: Used for the training window; the "or disposition" half must be filtered out
#: downstream by the extraction step.
ITEM_COMPLETED_ACQUISITION = "2.01"

#: 8-K item reporting entry into a material definitive agreement — where a deal
#: is *announced*. Used for the holdout window, because Item 2.01 lags
#: announcement by 1-6 months and would surface pre-cutoff deals.
ITEM_MATERIAL_AGREEMENT = "1.01"

#: 8-K item reporting financial statements and exhibits; press releases attach here.
ITEM_EXHIBITS = "9.01"

_ITEM_RE = re.compile(r"\b(\d{1,2}\.\d{2})\b")
_PRESS_RELEASE_TYPES = ("EX-99.1", "EX-99", "EX-99.2")
_PRESS_RELEASE_NAME_RE = re.compile(r"ex(?:hibit)?[-_ ]?99", re.IGNORECASE)
_TEXTUAL_SUFFIXES = (".htm", ".html", ".txt")
#: ``index.json`` puts the name of a display *icon* in its ``type`` field
#: ("text.gif", "image2.gif"), not the EDGAR document type. Verified against
#: live filings; see docs/plan.md. Such values are discarded.
_ICON_TYPE_RE = re.compile(r"\.(gif|png|jpe?g|svg)$", re.IGNORECASE)


def parse_date(value: str | date | None) -> date | None:
    """Parse an EDGAR ``YYYY-MM-DD`` date, tolerating empty strings.

    Args:
        value: An ISO date string, a ``date``, or an empty/None placeholder.

    Returns:
        The parsed date, or ``None`` if the value was blank.
    """
    if value is None or isinstance(value, date):
        return value
    text = value.strip()
    if not text:
        return None
    return date.fromisoformat(text[:10])


def parse_items(value: str | None) -> list[str]:
    """Extract 8-K item numbers from the submissions ``items`` field.

    The field is usually a comma-separated list like ``"2.01,9.01"``, but some
    rows carry prose. Pulling out anything shaped like an item number handles
    both without guessing at the delimiter.

    Args:
        value: Raw ``items`` string.

    Returns:
        Item numbers in order of appearance, deduplicated.
    """
    if not value:
        return []
    seen: dict[str, None] = {}
    for match in _ITEM_RE.findall(value):
        seen.setdefault(match, None)
    return list(seen)


# --------------------------------------------------------------------------- #
# Filing index
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilingRef:
    """One filing as listed in a company's submissions index.

    Attributes:
        cik: Zero-padded 10-digit CIK of the filer.
        accession: Accession number with dashes, e.g. ``0000320193-24-000081``.
        form: Form type, e.g. ``8-K``.
        filing_date: Date EDGAR accepted the filing. This is the date the
            document became public, and therefore the one the as-of guard uses.
        report_date: Period or event date reported, when given.
        items: 8-K item numbers, empty for other forms.
        primary_document: Filename of the main document.
        primary_description: EDGAR's description of the main document.
    """

    cik: str
    accession: str
    form: str
    filing_date: date
    report_date: date | None = None
    items: tuple[str, ...] = ()
    primary_document: str = ""
    primary_description: str = ""

    @property
    def accession_bare(self) -> str:
        """Accession number with dashes removed, as used in archive paths."""
        return self.accession.replace("-", "")

    def has_item(self, item: str) -> bool:
        """Return whether this filing reports a given 8-K item."""
        return item in self.items


def _column(table: Mapping[str, object], key: str, length: int) -> list[str]:
    """Return one column of a submissions table, padded to ``length``."""
    raw = table.get(key)
    values = [("" if v is None else str(v)) for v in raw] if isinstance(raw, list) else []
    if len(values) < length:
        values.extend([""] * (length - len(values)))
    return values


def iter_filings(payload: Mapping[str, object], cik: str | None = None) -> Iterator[FilingRef]:
    """Yield filings from a submissions payload or an older submissions page.

    The main ``submissions/CIK##########.json`` response nests its columns under
    ``filings.recent``; the overflow pages named in ``filings.files[].name`` hold
    the same columns at the top level. Both shapes are accepted so callers do not
    branch.

    Args:
        payload: Parsed JSON from either endpoint.
        cik: CIK to stamp on each row. Defaults to the payload's own ``cik``.

    Yields:
        One :class:`FilingRef` per filing, skipping rows with an unparseable
        filing date.
    """
    filings = payload.get("filings")
    if isinstance(filings, dict) and isinstance(filings.get("recent"), dict):
        table = filings["recent"]
    else:
        table = payload
    if not isinstance(table, dict):
        return

    raw_cik = cik or payload.get("cik") or ""
    padded = re.sub(r"\D", "", str(raw_cik)).zfill(10)

    accessions = table.get("accessionNumber")
    if not isinstance(accessions, list):
        return
    length = len(accessions)
    forms = _column(table, "form", length)
    filing_dates = _column(table, "filingDate", length)
    report_dates = _column(table, "reportDate", length)
    items = _column(table, "items", length)
    documents = _column(table, "primaryDocument", length)
    descriptions = _column(table, "primaryDocDescription", length)

    for i, accession in enumerate(accessions):
        filed = parse_date(filing_dates[i])
        if filed is None:
            continue
        yield FilingRef(
            cik=padded,
            accession=str(accession),
            form=forms[i],
            filing_date=filed,
            report_date=parse_date(report_dates[i]),
            items=tuple(parse_items(items[i])),
            primary_document=documents[i],
            primary_description=descriptions[i],
        )


def submissions_pages(payload: Mapping[str, object]) -> list[str]:
    """Return the names of older submissions pages referenced by a payload.

    A company's main submissions response holds only its most recent ~1,000
    filings; anything earlier lives in these pages. A 2019 training window needs
    them for prolific filers.
    """
    filings = payload.get("filings")
    if not isinstance(filings, dict):
        return []
    files = filings.get("files")
    if not isinstance(files, list):
        return []
    return [str(f["name"]) for f in files if isinstance(f, dict) and f.get("name")]


def filings_as_of(
    filings: Iterable[FilingRef],
    as_of: date,
    *,
    since: date | None = None,
) -> list[FilingRef]:
    """Filter filings to those public on or before ``as_of``.

    This is the as-of guard in its simplest form. Filtering on ``filing_date``
    rather than ``report_date`` is deliberate: a filing about a January event
    that was not filed until March was not knowable in February.

    Args:
        filings: Filings to filter.
        as_of: Inclusive upper bound on ``filing_date``.
        since: Optional inclusive lower bound.

    Returns:
        Matching filings, newest first.
    """
    kept = [
        f for f in filings if f.filing_date <= as_of and (since is None or f.filing_date >= since)
    ]
    return sorted(kept, key=lambda f: (f.filing_date, f.accession), reverse=True)


def select_8k_items(
    filings: Iterable[FilingRef],
    items: Sequence[str],
) -> list[FilingRef]:
    """Return the 8-K filings reporting any of the given item numbers."""
    wanted = set(items)
    return [f for f in filings if f.form.upper().startswith("8-K") and wanted & set(f.items)]


# --------------------------------------------------------------------------- #
# Company profile
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CompanyProfile:
    """Identity fields lifted from a submissions payload.

    Attributes:
        cik: Zero-padded 10-digit CIK.
        name: Registrant name.
        sic: Four-digit SIC code, empty if EDGAR has none.
        sic_description: EDGAR's label for that SIC code.
        state: Business-address state or country code.
        state_of_incorporation: State of incorporation.
        entity_type: EDGAR entity type, e.g. ``operating``.
        tickers: Exchange tickers.
        exchanges: Exchanges those tickers trade on.
        fiscal_year_end: ``MMDD`` fiscal year end, if given.
    """

    cik: str
    name: str
    sic: str = ""
    sic_description: str = ""
    state: str = ""
    state_of_incorporation: str = ""
    entity_type: str = ""
    tickers: tuple[str, ...] = ()
    exchanges: tuple[str, ...] = ()
    fiscal_year_end: str = ""

    @property
    def sic2(self) -> str:
        """First two SIC digits — the broad industry group."""
        return self.sic[:2]

    @property
    def sic3(self) -> str:
        """First three SIC digits."""
        return self.sic[:3]


def parse_company_profile(payload: Mapping[str, object]) -> CompanyProfile:
    """Build a :class:`CompanyProfile` from a submissions payload."""

    def text(key: str) -> str:
        value = payload.get(key)
        return "" if value is None else str(value).strip()

    addresses = payload.get("addresses")
    state = ""
    if isinstance(addresses, dict):
        business = addresses.get("business")
        if isinstance(business, dict):
            state = str(business.get("stateOrCountry") or "").strip()

    def string_tuple(key: str) -> tuple[str, ...]:
        value = payload.get(key)
        if not isinstance(value, list):
            return ()
        return tuple(str(v).strip() for v in value if v)

    return CompanyProfile(
        cik=re.sub(r"\D", "", text("cik")).zfill(10),
        name=text("name"),
        sic=text("sic"),
        sic_description=text("sicDescription"),
        state=state,
        state_of_incorporation=text("stateOfIncorporation"),
        entity_type=text("entityType"),
        tickers=string_tuple("tickers"),
        exchanges=string_tuple("exchanges"),
        fiscal_year_end=text("fiscalYearEnd"),
    )


# --------------------------------------------------------------------------- #
# Filing documents
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class FilingDocument:
    """One document inside a filing's archive directory.

    Attributes:
        name: Filename within the directory.
        doc_type: EDGAR document type, e.g. ``EX-99.1``. Often blank.
        size: Size in bytes as reported by the directory listing.
    """

    name: str
    doc_type: str = ""
    size: int = 0

    @property
    def is_textual(self) -> bool:
        """Return whether this document is worth extracting text from."""
        return self.name.lower().endswith(_TEXTUAL_SUFFIXES)


def parse_filing_index(payload: Mapping[str, object]) -> list[FilingDocument]:
    """Parse an archive directory ``index.json`` into document records."""
    directory = payload.get("directory")
    if not isinstance(directory, dict):
        return []
    items = directory.get("item")
    if not isinstance(items, list):
        return []
    documents: list[FilingDocument] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        raw_size = str(item.get("size") or "0")
        doc_type = str(item.get("type") or "").strip()
        documents.append(
            FilingDocument(
                name=str(item["name"]),
                doc_type="" if _ICON_TYPE_RE.search(doc_type) else doc_type,
                size=int(raw_size) if raw_size.isdigit() else 0,
            )
        )
    return documents


def find_press_releases(documents: Iterable[FilingDocument]) -> list[FilingDocument]:
    """Return the press-release exhibits in a filing, best candidate first.

    A deal 8-K's own body is usually two sentences of boilerplate pointing at the
    attached press release, which is where the target description actually lives.
    That makes ``EX-99.1`` the document extraction depends on.

    Matching prefers the declared document type and falls back to the filename.
    The fallback carries most of the weight in practice: ``index.json`` reports a
    display icon rather than a document type, so the authoritative type comes
    either from full-text search (see :class:`FullTextHit`) or not at all.
    """
    ranked: list[tuple[int, FilingDocument]] = []
    for doc in documents:
        if not doc.is_textual:
            continue
        upper = doc.doc_type.upper()
        if upper in _PRESS_RELEASE_TYPES:
            ranked.append((_PRESS_RELEASE_TYPES.index(upper), doc))
        elif not doc.doc_type and _PRESS_RELEASE_NAME_RE.search(doc.name):
            ranked.append((len(_PRESS_RELEASE_TYPES), doc))
    ranked.sort(key=lambda pair: (pair[0], pair[1].name))
    return [doc for _, doc in ranked]


#: Inline-XBRL primary documents are named ``<ticker>-<yyyymmdd>.htm`` by every
#: major filing agent, which is the most reliable way to spot the 8-K body in a
#: directory listing whose ``type`` column is useless.
_INLINE_XBRL_NAME_RE = re.compile(r"^[a-z0-9]{1,10}-20\d{6}\.htm$", re.IGNORECASE)
_FORM_8K_NAME_RE = re.compile(r"(^|[^a-z0-9])8[-_]?k([^a-z0-9]|$)", re.IGNORECASE)
_EXHIBIT_NAME_RE = re.compile(r"ex(?:hibit)?[-_ ]?\d", re.IGNORECASE)


def find_primary_document(documents: Iterable[FilingDocument]) -> FilingDocument | None:
    """Return the filing's own body document, or ``None`` if it cannot be told.

    Extraction reads the press release *and* the 8-K body: the body carries the
    item numbers and the signed-off legal framing, the exhibit carries the
    narrative. Neither alone is enough.

    Ranking, best first: a declared form type; the inline-XBRL ``ticker-date.htm``
    convention; a name mentioning 8-K; then the largest remaining non-exhibit
    document. Exhibits and XBRL sidecars are excluded throughout.
    """
    ranked: list[tuple[int, int, str, FilingDocument]] = []
    for doc in documents:
        name = doc.name
        if not doc.is_textual or _EXHIBIT_NAME_RE.search(name):
            continue
        if doc.doc_type.upper().startswith("8-K"):
            rank = 0
        elif _INLINE_XBRL_NAME_RE.match(name):
            rank = 1
        elif _FORM_8K_NAME_RE.search(name):
            rank = 2
        elif not doc.doc_type:
            rank = 3
        else:
            continue
        # Within a rank, prefer the larger document: the body, not a cover letter.
        ranked.append((rank, -doc.size, name, doc))
    if not ranked:
        return None
    ranked.sort(key=lambda row: row[:3])
    return ranked[0][3]


_SCRIPT_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_HIDDEN_RE = re.compile(
    r"<div[^>]*style=[\"'][^\"']*display:\s*none[^\"']*[\"'][^>]*>.*?</div>",
    re.IGNORECASE | re.DOTALL,
)
_BREAK_RE = re.compile(r"</(p|div|tr|h[1-6]|li|table)\s*>|<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_BLANKS_RE = re.compile(r"\n\s*\n\s*\n+")
_HTMLISH_RE = re.compile(r"<\s*(html|body|div|p|table|tr|td|font|span|br)\b", re.IGNORECASE)


def html_to_text(raw: str) -> str:
    """Convert a filing's HTML to plain text.

    Deliberately regex-based rather than a full HTML parser. Filing HTML is
    machine-generated and shallow, the output only has to be good enough for a
    model to read and for the verifier to substring-match claims against, and
    this avoids adding a parser dependency for one function.

    Inline XBRL ``<ix:header>`` blocks are dropped: they are hidden metadata that
    would otherwise appear as a wall of numbers at the top of every document.

    In HTML, a newline in the source is just whitespace, so hard-wrapped
    paragraphs are rejoined before block tags become real line breaks. Without
    that, a sentence wrapped at column 80 arrives split across lines and the
    verifier's substring match against a quoted claim fails. Plain-text exhibits
    are left alone, since there their newlines are the only structure present.
    """
    text = re.sub(r"<ix:header\b.*?</ix:header>", " ", raw, flags=re.IGNORECASE | re.DOTALL)
    if _HTMLISH_RE.search(text):
        text = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = _SCRIPT_RE.sub(" ", text)
    text = _HIDDEN_RE.sub(" ", text)
    text = _BREAK_RE.sub("\n", text)
    text = _TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("​", "")
    text = _WS_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _BLANKS_RE.sub("\n\n", text).strip()


# --------------------------------------------------------------------------- #
# XBRL company facts
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Concept:
    """A financial concept and the XBRL tags companies report it under.

    Attributes:
        name: Short key used by callers, e.g. ``revenue``.
        tags: Candidate us-gaap tags. Order breaks ties between tags reporting
            the same period; it does *not* short-circuit the search.
        annual_only: Restrict to full-year observations. True for flow concepts,
            where a quarterly figure is not comparable to an annual one.
    """

    name: str
    tags: tuple[str, ...]
    annual_only: bool = False


#: Concepts read from companyfacts, with their tag aliases.
#:
#: Revenue needs four aliases because ASC 606 moved most filers off ``Revenues``
#: around 2018. Crucially, many kept the old tag populated with stale data:
#: Thermo Fisher's ``Revenues`` series stops at FY2017 (\$20.9bn) while its
#: ``RevenueFromContractWithCustomer...`` series runs to FY2025 (\$44.6bn).
#: Searching aliases in order and stopping at the first that has *any* data
#: therefore returns an eight-year-stale figure, so every alias is searched and
#: the latest period wins.
CONCEPTS: dict[str, Concept] = {
    "revenue": Concept(
        "revenue",
        (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
        annual_only=True,
    ),
    "assets": Concept("assets", ("Assets",)),
    "cash": Concept(
        "cash",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
    ),
    "debt": Concept("debt", ("LongTermDebtNoncurrent", "LongTermDebt")),
}

#: Tag aliases per concept, for callers that only need the names.
FACT_TAGS: dict[str, tuple[str, ...]] = {k: v.tags for k, v in CONCEPTS.items()}


@dataclass(frozen=True, slots=True)
class Fact:
    """One XBRL fact, with the dates needed to reason about knowability.

    Attributes:
        tag: The us-gaap tag it came from.
        value: Reported numeric value.
        end: Period end the value describes.
        filed: Date the filing carrying it became public. The as-of guard filters
            on this, not ``end``.
        start: Period start, for flow concepts like revenue. ``None`` for
            instantaneous concepts like total assets.
        form: Form the value was reported on.
        unit: Unit of measure, e.g. ``USD``.
        fiscal_period: ``fp`` field, e.g. ``FY`` or ``Q3``.
    """

    tag: str
    value: float
    end: date
    filed: date
    start: date | None = None
    form: str = ""
    unit: str = "USD"
    fiscal_period: str = ""
    accession: str = ""  # added in this project: cite the filing a figure came from

    @property
    def is_annual(self) -> bool:
        """Return whether this fact covers a full year.

        ``companyfacts`` interleaves quarterly and annual observations of the same
        tag. Taking the latest one as a size proxy silently mixes a single quarter
        into a series of full years, so revenue is filtered to annual durations.
        The period span is the reliable test; ``fp`` is the fallback for facts
        that omit a start date.
        """
        if self.start is not None:
            return 350 <= (self.end - self.start).days <= 380
        return self.fiscal_period.upper() == "FY"


def iter_facts(
    companyfacts: Mapping[str, object],
    tag: str,
    *,
    unit: str = "USD",
    taxonomy: str = "us-gaap",
) -> Iterator[Fact]:
    """Yield every reported value of one XBRL tag."""
    facts = companyfacts.get("facts")
    if not isinstance(facts, dict):
        return
    taxonomy_facts = facts.get(taxonomy)
    if not isinstance(taxonomy_facts, dict):
        return
    entry = taxonomy_facts.get(tag)
    if not isinstance(entry, dict):
        return
    units = entry.get("units")
    if not isinstance(units, dict):
        return
    observations = units.get(unit)
    if not isinstance(observations, list):
        return
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        end = parse_date(obs.get("end"))
        filed = parse_date(obs.get("filed"))
        value = obs.get("val")
        if end is None or filed is None or not isinstance(value, int | float):
            continue
        yield Fact(
            tag=tag,
            value=float(value),
            end=end,
            filed=filed,
            start=parse_date(obs.get("start")),
            form=str(obs.get("form") or ""),
            unit=unit,
            fiscal_period=str(obs.get("fp") or ""),
            accession=str(obs.get("accn") or ""),
        )


def latest_fact_as_of(
    companyfacts: Mapping[str, object],
    concept: str,
    as_of: date,
    *,
    unit: str = "USD",
) -> Fact | None:
    """Return the most recent value of a concept that was public by ``as_of``.

    The filter is on ``filed``, not ``end``. A fiscal year ending in December is
    not knowable until the 10-K is filed in February, so a feature computed for a
    January announcement date must use the *prior* year's figure. Getting this
    backwards is the single easiest way to leak the future into the model, which
    is why it lives in one function with a test on it.

    Args:
        companyfacts: Parsed ``companyfacts`` payload.
        concept: Key in :data:`FACT_TAGS`, or a raw us-gaap tag.
        as_of: Inclusive cutoff on the filing date.
        unit: XBRL unit to read.

    Returns:
        The latest knowable fact, or ``None`` if the company never reported the
        concept before that date.
    """
    definition = CONCEPTS.get(concept) or Concept(concept, (concept,))
    best: Fact | None = None
    for tag in definition.tags:
        for fact in iter_facts(companyfacts, tag, unit=unit):
            if fact.filed > as_of:
                continue
            if definition.annual_only and not fact.is_annual:
                continue
            # Latest period end wins, then the later filing (a restatement).
            # Strict comparison means an earlier alias keeps a tie, which is why
            # CONCEPTS lists the modern tag first.
            if best is None or (fact.end, fact.filed) > (best.end, best.filed):
                best = fact
    return best


@dataclass(frozen=True, slots=True)
class Financials:
    """Size and balance-sheet proxies for one company as of a date.

    Attributes:
        as_of: The date these values were knowable on.
        revenue: Latest reported revenue, if any.
        assets: Latest reported total assets.
        cash: Latest reported cash and equivalents.
        debt: Latest reported long-term debt.
        facts: The underlying facts, keyed by concept, for provenance.
    """

    as_of: date
    revenue: float | None = None
    assets: float | None = None
    cash: float | None = None
    debt: float | None = None
    facts: dict[str, Fact] = field(default_factory=dict)

    @property
    def size_proxy(self) -> float | None:
        """Revenue if reported, else total assets.

        Revenue is the better comparability measure across industries, but
        financial-adjacent filers often report assets only.
        """
        return self.revenue if self.revenue is not None else self.assets

    @property
    def cash_to_assets(self) -> float | None:
        """Cash as a fraction of total assets — a dry-powder proxy."""
        if self.cash is None or not self.assets:
            return None
        return self.cash / self.assets


def financials_as_of(companyfacts: Mapping[str, object], as_of: date) -> Financials:
    """Assemble every size and balance-sheet proxy knowable at ``as_of``."""
    facts: dict[str, Fact] = {}
    for concept in ("revenue", "assets", "cash", "debt"):
        fact = latest_fact_as_of(companyfacts, concept, as_of)
        if fact is not None:
            facts[concept] = fact
    return Financials(
        as_of=as_of,
        revenue=facts["revenue"].value if "revenue" in facts else None,
        assets=facts["assets"].value if "assets" in facts else None,
        cash=facts["cash"].value if "cash" in facts else None,
        debt=facts["debt"].value if "debt" in facts else None,
        facts=facts,
    )


# --------------------------------------------------------------------------- #
# Full-text search
# --------------------------------------------------------------------------- #

_DISPLAY_NAME_RE = re.compile(r"\s*\((?:CIK\s*)?[^()]*\)\s*$")
_TICKER_RE = re.compile(r"\(([A-Z][A-Z0-9.\-]{0,5}(?:,\s*[A-Z][A-Z0-9.\-]{0,5})*)\)")


def clean_display_name(display_name: str) -> tuple[str, tuple[str, ...]]:
    """Split an EDGAR search display name into a company name and its tickers.

    Search results label companies as
    ``"Embecta Corp.  (EMBC)  (CIK 0001872789)"``.

    Args:
        display_name: The raw ``display_names`` entry.

    Returns:
        A ``(name, tickers)`` pair.
    """
    tickers: tuple[str, ...] = ()
    without_cik = _DISPLAY_NAME_RE.sub("", display_name).strip()
    match = _TICKER_RE.search(without_cik)
    if match:
        tickers = tuple(t.strip() for t in match.group(1).split(",") if t.strip())
        without_cik = without_cik[: match.start()].strip()
    return without_cik.strip(), tickers


@dataclass(frozen=True, slots=True)
class FullTextHit:
    """One document matched by EDGAR full-text search.

    Search hits are at *document* level, not filing level, and each carries the
    filer's identity alongside the document's own type. That makes one search
    request stand in for what would otherwise be a submissions lookup plus a
    directory listing plus a document-type lookup per filing.

    Attributes:
        accession: Accession number of the containing filing.
        cik: Zero-padded 10-digit CIK of the filer.
        company_name: Registrant name, with ticker and CIK suffixes stripped.
        tickers: Tickers parsed out of the display name.
        document: Filename of the matched document within the filing.
        doc_type: EDGAR document type, e.g. ``EX-99.1``.
        doc_description: Filer-supplied description, e.g. ``PRESS RELEASE``.
        form: Form type of the containing filing.
        filing_date: Date the filing became public.
        items: 8-K item numbers on the containing filing.
        sic: Filer's SIC code.
        business_state: Business-address state.
        incorporation_state: State of incorporation.
        score: Relevance score, for diagnostics only.
    """

    accession: str
    cik: str
    company_name: str
    document: str
    doc_type: str = ""
    doc_description: str = ""
    form: str = ""
    filing_date: date | None = None
    items: tuple[str, ...] = ()
    tickers: tuple[str, ...] = ()
    sic: str = ""
    business_state: str = ""
    incorporation_state: str = ""
    score: float = 0.0

    @property
    def is_press_release(self) -> bool:
        """Return whether this document is an EX-99 press-release exhibit."""
        return self.doc_type.upper().startswith("EX-99")


def full_text_total(payload: Mapping[str, object]) -> int:
    """Return the total number of documents matching a search.

    This is the true total, which may exceed the number of hits any one response
    carries and may exceed what paging can reach. Callers partition by date to
    stay inside the reachable window.
    """
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return 0
    total = hits.get("total")
    if isinstance(total, dict):
        return int(total.get("value") or 0)
    return int(total or 0)


def parse_full_text_search(payload: Mapping[str, object]) -> list[FullTextHit]:
    """Parse an EDGAR full-text search response into document hits."""
    hits = payload.get("hits")
    if not isinstance(hits, dict):
        return []
    rows = hits.get("hits")
    if not isinstance(rows, list):
        return []

    results: list[FullTextHit] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        source = row.get("_source")
        if not isinstance(source, dict):
            continue

        # The document id is "<accession>:<filename>".
        identifier = str(row.get("_id") or "")
        accession, _, document = identifier.partition(":")
        if not accession:
            continue

        def first(key: str, src: Mapping[str, object] = source) -> str:
            value = src.get(key)
            if isinstance(value, list):
                return str(value[0]).strip() if value else ""
            return "" if value is None else str(value).strip()

        def tuple_of(key: str, src: Mapping[str, object] = source) -> tuple[str, ...]:
            value = src.get(key)
            if not isinstance(value, list):
                return ()
            return tuple(str(v).strip() for v in value if str(v).strip())

        name, tickers = clean_display_name(first("display_names"))
        raw_cik = first("ciks")
        results.append(
            FullTextHit(
                accession=str(source.get("adsh") or accession),
                cik=re.sub(r"\D", "", raw_cik).zfill(10) if raw_cik else "",
                company_name=name,
                tickers=tickers,
                document=document,
                doc_type=first("file_type"),
                doc_description=first("file_description"),
                form=first("form") or first("root_forms"),
                filing_date=parse_date(first("file_date")),
                items=tuple_of("items"),
                sic=first("sics"),
                business_state=first("biz_states"),
                incorporation_state=first("inc_states"),
                score=float(row.get("_score") or 0.0),
            )
        )
    return results


def hits_as_of(hits: Iterable[FullTextHit], as_of: date) -> list[FullTextHit]:
    """Filter search hits to documents public on or before ``as_of``.

    Search requests already carry a date range, but the guard is applied to the
    results as well. A request built with the wrong bound, or a cached response
    from a wider earlier query, would otherwise leak future filings into a
    backtest, and this is the last point before that data reaches a model.
    """
    return [h for h in hits if h.filing_date is not None and h.filing_date <= as_of]
