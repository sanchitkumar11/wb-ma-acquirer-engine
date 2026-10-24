"""Streamlit view of the engine's output: `streamlit run app.py`.

Reads only from the cache (LLM_PROVIDER is forced to replay), so it runs with no key and costs
nothing.
The CLI (`python -m acquirer_engine run`) is the primary path and writes the same content as
Markdown.
Layout follows the reference decks through structure, not styling: action titles as headers, a
criteria table with "what does that mean?", a sponsor scorecard, risks as concern / mitigant /
reality
columns, a football field per buyer, and source and as-of footnotes on every page.
"""

from __future__ import annotations

import json
import os
import re

from acquirer_engine.feedback import ACTION_WORDS
from acquirer_engine.render import CITATION_LEGEND
from acquirer_engine.wording import CSV_ONLY_NOTE

os.environ["LLM_PROVIDER"] = "replay"  # before any engine import: the app never makes a live call

from typing import TYPE_CHECKING

import altair as alt
import pandas as pd
import streamlit as st

from acquirer_engine.charts import buyer_price_ranges
from acquirer_engine.compare import comparison_note, shared_rows, unique_rows
from acquirer_engine.config import OWNERSHIP_TYPES, REGIONS, Target, load_target
from acquirer_engine.errors import NotCachedError, PaidCallNotAllowedError
from acquirer_engine.feedback import (
    REASONS,
    accept,
    active_flags,
    learned_rules,
    record,
    suggestions,
)
from acquirer_engine.figures import (
    buyer_map,
    compare_map,
    football_field_figure,
    funnel_figure,
    rings_figure,
)
from acquirer_engine.listbuild import suggested_moves
from acquirer_engine.onepager import CITATION, OnePager
from acquirer_engine.paths import ROOT
from acquirer_engine.render import NOT_RESEARCHED, SECTIONS, key_numbers, ordinal
from acquirer_engine.report import build
from acquirer_engine.rings import GROUP
from acquirer_engine.summary import (
    METHOD_NOTE,
    call_list,
    headline_text,
    motive_rows,
    not_researched,
    recommended_approach,
    review_log_rows,
    sources_rows,
    sponsor_scorecard,
    target_line,
    tier2_groups,
)
from acquirer_engine.wording import plain_terms

if TYPE_CHECKING:
    from collections.abc import Callable

    from acquirer_engine.report_types import Report

NAVY, ACCENT, GREY = "#1B2A4A", "#0E7C86", "#3C4650"
st.set_page_config(page_title="Likely acquirers", layout="wide")
st.markdown(
    f"<style>h1, h2, h3, h4, h5 {{color: {NAVY} !important;}}</style>", unsafe_allow_html=True
)  # the one style rule


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Scoring buyers...")
def get_report(target: Target, csv_only: bool, feedback_state: str) -> Report:
    """Return the report for a target, from the cache only (cached per target and per feedback
    state:
    a new flag or an accepted rule changes feedback_state, so the list is rebuilt)."""
    return build(target, csv_only=csv_only, replay=True)


def feedback_state(target: Target) -> str:
    """Return the target's active flags and accepted rules as text: the app's cache key for them."""
    return json.dumps([active_flags(target), learned_rules(target.sector)], sort_keys=True)


def flag_editor(buyers: list[str], key: str) -> None:
    """A "flag as irrelevant" control on every row: tick, pick a reason, add a note, save.

    Saved flags go to feedback/feedback.jsonl and apply on the next run (here: immediately, from
    the cache; a refilled Tier 1 buyer's page needs a CLI run with --allow-paid).
    """
    with st.expander("Flag a buyer as irrelevant"):
        rows = pd.DataFrame(
            {"Buyer": buyers, "Flag as irrelevant": False, "Reason": REASONS[0], "Note": ""}
        )
        edited = st.data_editor(
            rows,
            hide_index=True,
            width="stretch",
            key=f"flags-{key}",
            disabled=["Buyer"],
            column_config={
                "Reason": st.column_config.SelectboxColumn(options=list(REASONS), required=True),
                "Note": st.column_config.TextColumn(help="Required for 'other'"),
            },
        )
        if st.button("Save flags", key=f"save-{key}"):
            chosen = edited[edited["Flag as irrelevant"]]
            try:
                for r in chosen.itertuples():
                    record(TARGET, r.Buyer, r.Reason, r.Note or "")
            except ValueError as exc:
                st.error(str(exc))
                return
            st.rerun()


def md(text: str) -> str:
    """Return text safe for st.markdown: a bare $ would otherwise start a LaTeX formula."""
    return text.replace("$", "\\$")


def table(df: pd.DataFrame) -> None:
    """Show a wrapping table; st.table renders cells as Markdown, so text cells are escaped too."""
    st.table(df.map(lambda v: md(v) if isinstance(v, str) else v))


def cite(text: str) -> str:
    """Return markdown-safe text with [fact ID] citations shown small and grey."""
    return CITATION.sub(lambda m: f" :gray[[{m.group(1)}]]", md(plain_terms(text)))


def footer(extra: str = "") -> None:
    """Print the as-of footnote every page carries."""
    st.divider()
    st.caption(
        md(
            f"{TARGET.data_note()}. Deal figures come from the provided CSV; real-world facts from "
            "dated, sourced buyer "
            f"profiles (see Sources). {extra}"
        )
    )


def target_form() -> tuple[Target, bool]:
    """Sidebar form for the target profile (stretch goal: arbitrary targets).

    Defaults come from a sample target file (configs/target_*.yaml; APP_TARGET picks the first one
    shown, for tests); the form then edits any field.
    """
    samples = sorted(
        (ROOT / "configs").glob("target_*.yaml"), key=lambda f: f.stem != "target_default"
    )
    first = next((i for i, f in enumerate(samples) if f.name == os.environ.get("APP_TARGET")), 0)
    chosen = st.sidebar.selectbox(
        "Start from a sample target", samples, index=first, format_func=lambda f: f.stem[7:]
    )
    base = load_target(chosen)
    with st.sidebar.form(
        f"target-{chosen.stem}"
    ):  # one form per sample, so its fields reset to that sample
        st.markdown("**Target profile**")
        sector = st.selectbox("Sub-sector", sorted(GROUP), index=sorted(GROUP).index(base.sector))
        ev = st.number_input(
            "Enterprise value ($M)", 20.0, 5000.0, base.enterprise_value_mm, step=10.0
        )
        region = st.selectbox("Region", sorted(REGIONS), index=sorted(REGIONS).index(base.region))
        ownership = st.selectbox(
            "Ownership",
            sorted(OWNERSHIP_TYPES),
            index=sorted(OWNERSHIP_TYPES).index(base.ownership),
        )
        margin = st.number_input("EBITDA margin (%)", 0.5, 60.0, base.ebitda_margin_pct, step=0.5)
        growth = st.number_input(
            "Revenue growth (%)", -20.0, 80.0, base.revenue_growth_pct, step=0.5
        )
        csv_only = st.checkbox("CSV only (no buyer profiles)", value=False)
        st.form_submit_button("Run")
    st.sidebar.caption(
        f"{base.data_note()}. One-pagers are shown only for targets already generated by the CLI."
    )
    return Target(sector, ev, region, ownership, margin, growth, base.as_of), csv_only


try:  # an out-of-range profile or too few comparable deals: say why, instead of a stack trace
    TARGET, CSV_ONLY = target_form()
    REP = get_report(TARGET, CSV_ONLY, feedback_state(TARGET))
except ValueError as exc:
    st.error(f"This target can't be scored: {exc}")
    st.stop()
PAGES = {r["buyer"]: OnePager.model_validate(r["page"]) for r in (REP.results or [])}


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


STATIC = {"staticPlot": True}  # the summary charts are slides, not dashboards


def summary_page() -> None:
    """Page one, the MD's one-pager: headline, buyer map beside the summary, call list, where the
    ask sits."""
    es = REP.exec_summary
    st.markdown(f"### {md(headline_text(REP))}")  # smaller than a page title: it is a sentence
    if REP.res.review.csv_only:
        st.warning(CSV_ONLY_NOTE)
    if REP.results is None:
        st.info(
            "One-pagers have not been generated for this target; the angles below are built from "
            "the data. To write the "
            "pages, save the profile as a YAML and run `python -m acquirer_engine run --target "
            "<file> --allow-paid`."
        )
    left, right = st.columns([65, 35])
    left.plotly_chart(buyer_map(REP), config=STATIC, width="stretch")
    with right:
        st.markdown("#### Executive summary")
        st.markdown("\n".join(f"- {md(plain_terms(x))}" for x in es["sentences"]))
        if not REP.exec_from_llm:
            st.caption(
                "Template summary: the LLM-written version for this target is not in the cache."
            )
        st.markdown("#### Recommended approach")
        st.markdown("\n".join(f"{i}. {md(x)}" for i, x in enumerate(recommended_approach(REP), 1)))
    st.markdown("#### Tier 1: who to call, in order")
    calls = pd.DataFrame(call_list(REP)).astype(
        {"Priority": str}
    )  # a text index: st.table hides a numeric one
    table(calls.set_index("Priority"))
    flag_editor(REP.tier1, "tier1")
    if REP.res.user_flags:
        with st.expander(f"Deal-team flags and accepted moves ({len(REP.res.user_flags)})"):
            for buyer, line in REP.res.user_flags.items():
                left_col, right_col = st.columns([5, 1])
                verb = ACTION_WORDS[line["action"]]
                left_col.markdown(
                    md(
                        f"**{buyer}** {verb}: {line['reason']}"
                        + (f" ({line['note']})" if line["note"] else "")
                        + f", {line['timestamp'][:10]}"
                    )
                )
                if right_col.button("Unflag", key=f"unflag-{buyer}"):
                    record(TARGET, buyer, action="unflag")
                    st.rerun()
    st.plotly_chart(football_field_figure(REP), config=STATIC, width="stretch")
    st.divider()
    st.caption(md(target_line(REP)))


def bench_page() -> None:
    """Tier 2 in four groups, each with what would promote a buyer, then the sponsor scorecard."""
    st.title("The bench: Tier 2 buyers, by what holds them back")
    for g in tier2_groups(REP):
        st.markdown(f"#### {g['name']}")
        st.caption(md(f"What would promote one: {g['promote']}."))
        table(pd.DataFrame(g["rows"]).set_index("Buyer"))
        flag_editor([r["Buyer"] for r in g["rows"]], slug(g["name"]))
    st.markdown(
        "#### Sponsor scorecard: the file barely separates sponsors, so compare them side by side"
    )
    st.dataframe(pd.DataFrame(sponsor_scorecard(REP)), hide_index=True, width="stretch")
    footer()


def built_page() -> None:
    """How the list was built: stages, rings, motives, review log, method note."""
    st.title("How we built the list")
    st.plotly_chart(
        funnel_figure(REP), config=STATIC, width="stretch"
    )  # full width: side by side cuts the reasons off
    st.plotly_chart(rings_figure(REP), config=STATIC, width="stretch")
    st.markdown("#### Buyer motives and the evidence for them")
    table(pd.DataFrame(motive_rows(REP)).drop(columns=["_share"]).set_index("Motive"))
    learning_section()
    suggested_moves_section()
    st.markdown("#### Review log: model rank vs final tier, with the reason for every move")
    st.dataframe(
        pd.DataFrame(review_log_rows(REP)),
        hide_index=True,
        width="stretch",
        column_config={"Source": st.column_config.TextColumn(width="medium")},
    )
    footer(METHOD_NOTE)


def learning_section() -> None:
    """Rules suggested by the deal team's flags, each with its check against history and an accept
    button."""
    found = suggestions(
        TARGET, REP.res.user_flags, REP.res.deals, REP.res.profiles, REP.res.review.flags
    )
    accepted = learned_rules(TARGET.sector)
    if not (found or accepted or REP.res.user_flags):
        return
    st.markdown("#### Learning from the deal team's flags")
    st.caption(
        "A rule is suggested when two or more flags share a reason and the flagged buyers share a "
        "property a rule "
        "can act on. Nothing changes until you accept; conflicts and 'other' are about one deal "
        "and never become rules."
    )
    for i, s in enumerate(found):
        with st.container(border=True):
            st.markdown(md(f"**{s['label']}**"))
            st.caption(
                md(
                    f"From flags on {', '.join(s['from_flags'])}. Check against history: "
                    f"{s['history']}."
                )
            )
            if st.button("Accept this rule", key=f"accept-{i}"):
                accept(s)
                st.rerun()
    for r in accepted:
        st.markdown(md(f"- Accepted {r['accepted'][:10]}: {r['rule']} ({r['history']})"))


def suggested_moves_section() -> None:
    """Opt-in: the LLM reviewer's suggested moves on top of the rules' list, each with its reason
    code and
    cited facts. Accept or reject goes through the feedback loop; nothing is applied by itself."""
    st.markdown("#### Suggested moves from an LLM reviewer (opt-in)")
    if not st.toggle("Show suggested moves", key="show-suggestions"):
        st.caption(
            "The list above is built by rules alone. An LLM reviewer can suggest footprint, "
            "archetype or strategy "
            "moves with cited facts; the back-test can't score those judgments, so they are "
            "offered, not applied."
        )
        return
    try:
        out = suggested_moves(REP.res)
    except (NotCachedError, PaidCallNotAllowedError):
        st.caption(
            "No cached review for this list: run `python -m acquirer_engine review --suggest "
            "--allow-paid` for this target."
        )
        return
    if not out["moves"]:
        st.caption(
            "No open suggestions"
            + (f" ({out['rejected']} refused for missing evidence)." if out["rejected"] else ".")
        )
    for i, m in enumerate(out["moves"]):
        with st.container(border=True):
            st.markdown(
                md(f"**{m['action'].capitalize()} {m['buyer']}** ({m['code']}): {m['reason']}")
            )
            st.markdown("\n".join(md(f"- [{fid}] {text}") for fid, text in m["facts"]))
            left, right, _ = st.columns([1, 1, 4])
            cited = "; ".join(f"[{fid}] {text}" for fid, text in m["facts"])
            if left.button("Accept", key=f"accept-move-{i}"):
                record(
                    TARGET, m["buyer"], m["code"], f"{m['reason']} Evidence: {cited}", m["action"]
                )
                st.rerun()
            if right.button("Reject", key=f"reject-move-{i}"):
                record(TARGET, m["buyer"], f"{m['action']}:{m['code']}", m["reason"], "dismiss")
                st.rerun()


def chart(buyer: str) -> None:
    """Football field: horizontal ranges with the ask as a vertical line."""
    rows, ask, note = buyer_price_ranges(REP, buyer)
    base = alt.Chart(rows).encode(
        y=alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=320))
    )
    bars = base.mark_bar(color=ACCENT, height=18, cornerRadius=4).encode(
        x=alt.X("low:Q", title="Enterprise value ($M)", scale=alt.Scale(zero=False, nice=True)),
        x2="high:Q",
        tooltip=[
            alt.Tooltip("label:N", title="Range"),
            alt.Tooltip("low:Q", title="Low ($M)", format=",.0f"),
            alt.Tooltip("high:Q", title="High ($M)", format=",.0f"),
            alt.Tooltip("detail:N", title="Basis"),
        ],
    )
    ends = base.mark_text(align="left", dx=6, color=GREY).encode(
        x="high:Q", text=alt.Text("high:Q", format="$,.0f")
    )
    rule = (
        alt.Chart(pd.DataFrame({"ask": [ask]}))
        .mark_rule(color=NAVY, strokeWidth=2, strokeDash=[4, 3])
        .encode(x="ask:Q", tooltip=[alt.Tooltip("ask:Q", title="Ask ($M)", format=",.0f")])
    )
    st.altair_chart((bars + ends + rule).properties(height=60 * len(rows) + 40), width="stretch")
    st.caption(md(f"Dashed line: the ${ask:.0f}M ask. {note}"))
    with st.expander("Table view"):
        st.dataframe(rows, hide_index=True)


def buyer_page(buyer: str) -> None:
    pack, page = REP.packs[buyer], PAGES.get(buyer)
    rk = pack["ranking"]
    st.title(buyer)
    st.caption(
        f"{pack['buyer']['type']} | Tier 1, priority #{rk['position_on_list']} of 10 | ranked "
        f"{ordinal(rk['overall_rank'])} of "
        f"{rk['buyers_scored']} buyers | likelihood {rk['likelihood']} | conviction "
        f"{pack['conviction']['level']} | {TARGET.data_note()}"
    )
    if page:
        st.subheader(cite(page.headline))
    tiles = key_numbers(pack)
    for col, (label, value) in zip(
        st.columns(len(tiles)), tiles, strict=True
    ):  # small label over a bold value: no truncation
        col.caption(label)
        col.markdown(f"**{md(value)}**")
    if pack["profile_not_researched"] and not REP.res.review.csv_only:
        st.warning(NOT_RESEARCHED)
    if page is None:
        st.info(
            "This one-pager is not in the cache for this target. Key numbers, precedents and the "
            "football field are shown."
        )
    else:
        with st.container(border=True):
            st.markdown(f"**What sets it apart.** {cite(page.differentiator)}")
        for row in (SECTIONS[:2], SECTIONS[2:]):
            for col, (field, label) in zip(st.columns(2), row, strict=True):
                box = getattr(page, field)
                with col.container(border=True):
                    st.markdown(f"#### {md(plain_terms(box.title))}")
                    st.caption(label)
                    st.markdown("\n".join(f"- {cite(p)}" for p in box.points))
    st.markdown("#### Precedent activity in the file")
    prec = pd.DataFrame(pack["precedents"]).rename(
        columns={
            "id": "Deal ID",
            "buyer_as": "Bought as",
            "target": "Target",
            "year": "Year",
            "sub_sector": "Sub-sector",
            "fit": "Fit with the target",
            "size": "Size",
            "deal_type": "Type",
            "region": "Region",
            "ev_ebitda": "EV/EBITDA",
            "ev_revenue": "EV/Revenue",
            "outcome": "Outcome",
        }
    )
    if (prec["Bought as"] == pack["buyer"]["name"]).all():
        prec = prec.drop(columns=["Bought as"])
    st.dataframe(prec, hide_index=True, width="stretch")
    st.markdown(
        "#### Valuation: where the ask sits against what this buyer and the market have paid"
    )
    chart(buyer)
    if page:
        st.markdown("#### Risk flags")
        heads = st.columns([1, 2, 2, 2])
        for col, h in zip(heads, ["Risk", "Concern", "Mitigant", "Reality"], strict=True):
            col.markdown(f"**{h}**")
        for r in page.risk_flags:
            cols = st.columns([1, 2, 2, 2])
            cols[0].markdown(f"**{md(plain_terms(r.title))}**")
            for col, text in zip(cols[1:], (r.concern, r.mitigant, r.reality), strict=True):
                with col.container(border=True):
                    st.markdown(cite(text))
        with st.container(border=True):
            st.markdown(f"#### Conviction: {md(page.conviction.level)}")
            st.markdown(cite(page.conviction.rationale))
    with st.popover("Flag as irrelevant"):
        reason = st.selectbox("Reason", REASONS, key=f"reason-{buyer}")
        note = st.text_input("Note", key=f"note-{buyer}", help="Required for 'other'")
        if st.button("Save flag", key=f"flag-{buyer}"):
            try:
                record(TARGET, buyer, reason, note)
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    facts = "; ".join(
        f"[{f['id']}] {f['source']} (as of {f['as_of']})" for f in pack["profile_facts"]
    )
    footer(("Profile sources: " + facts + ". " if facts else "") + CITATION_LEGEND)


def sources_page() -> None:
    st.title("Sources")
    st.subheader(
        "Every real-world fact behind the review, with its source and the date it became true or "
        "public"
    )
    rows = pd.DataFrame(sources_rows(REP))
    if not_researched(REP) and not REP.res.review.csv_only:
        st.warning(
            f"Not researched (no sourced profile facts; deal-file evidence only): "
            f"{', '.join(not_researched(REP))}."
        )
    if rows.empty:
        st.info("CSV-only mode: no buyer profiles loaded.")
    else:
        rows = rows.rename(
            columns={
                "buyer": "Buyer",
                "kind": "Fact",
                "value": "Value",
                "as_of": "As of",
                "source": "Source",
                "accession": "EDGAR accession",
            }
        )
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            height=38 * len(rows) + 40,
            column_config={"Source": st.column_config.LinkColumn("Source")},
        )
    footer("EDGAR facts carry the filing's accession number.")


def compare_page() -> None:
    """Two sample targets side by side: the note, one buyer map with both Tier 1 lists, shared and
    unique buyers."""
    st.title("Compare two targets")
    samples = sorted((ROOT / "configs").glob("target_*.yaml"))
    names = [f.stem[7:] for f in samples]
    label = {
        n: (lambda t: f"{t.sector}, ${t.enterprise_value_mm:.0f}M, {t.region}")(load_target(f))
        for n, f in zip(names, samples, strict=True)
    }
    left, right = st.columns(2)
    pick_a = left.selectbox(
        "Target A",
        names,
        index=names.index("healthit") if "healthit" in names else 0,
        format_func=label.get,
    )
    pick_b = right.selectbox(
        "Target B",
        names,
        index=names.index("dental") if "dental" in names else min(1, len(names) - 1),
        format_func=label.get,
    )
    ta, tb = (load_target(ROOT / "configs" / f"target_{n}.yaml") for n in (pick_a, pick_b))
    a, b = get_report(ta, False, feedback_state(ta)), get_report(tb, False, feedback_state(tb))
    note, from_llm = comparison_note(
        a, b
    )  # the app is replay-only: the cached note, or the template
    st.markdown(f"### {md(note['headline'])}")
    st.markdown("\n".join(f"- {md(x)}" for x in note["sentences"]))
    if not from_llm:
        st.caption(
            "Template note: run `python -m acquirer_engine compare <a.yaml> <b.yaml> --allow-paid` "
            "to write this pair's note."
        )
    st.plotly_chart(compare_map(a, b), config=STATIC, width="stretch")
    st.markdown("#### Shared Tier 1 buyers, and the conviction points that differ")
    rows = [
        {
            "Buyer": r["buyer"],
            "Priority A / B": f"{r['priority_a']} / {r['priority_b']}",
            "Conviction A": r["conviction_a"],
            "Conviction B": r["conviction_b"],
            "Points that differ": "; ".join(
                f"{d['point']} (A: {d['a']}; B: {d['b']})" for d in r["points_that_differ"]
            )
            or "none",
        }
        for r in shared_rows(a, b)
    ]
    if rows:
        table(pd.DataFrame(rows).set_index("Buyer"))
    else:
        st.info("No buyer is on both Tier 1 lists.")
    for rep, other, heading in (
        (a, b, f"Only on A ({label[pick_a]})"),
        (b, a, f"Only on B ({label[pick_b]})"),
    ):
        st.markdown(f"#### {md(heading)}")
        rows = [
            {
                "Buyer": r["buyer"],
                "Type": r["type"],
                "Conviction": r["conviction"],
                "Why them": r["why_them"],
                "On the other list": r["in_other_list"],
            }
            for r in unique_rows(rep, other)
        ]
        if rows:  # the same target twice, or identical lists, leaves nothing unique
            table(pd.DataFrame(rows).set_index("Buyer"))
        else:
            st.info("Every buyer here is on the other list too.")
    footer()


def slug(name: str) -> str:
    """Return a URL-safe page path for a buyer, e.g. "Hellman & Friedman" -> "hellman-friedman"."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


START = os.environ.get(
    "APP_START_PAGE", "summary"
)  # which page opens first (lets tests render every page)


def page(fn: Callable[[], None], title: str, path: str) -> st.Page:
    """Return a navigation page; the one named by APP_START_PAGE opens first."""
    return st.Page(fn, title=title, url_path=path, default=(path == START))


nav = st.navigation(
    {
        "The list": [
            page(summary_page, "Summary", "summary"),
            page(bench_page, "Bench", "bench"),
            page(built_page, "How we built the list", "how-we-built-the-list"),
        ],
        "Tier 1 buyers": [
            page(lambda b=b: buyer_page(b), f"{i}. {b}", slug(b))
            for i, b in enumerate(REP.tier1, 1)
        ],
        "Reference": [page(sources_page, "Sources", "sources")],
        "Compare": [page(compare_page, "Compare targets", "compare")],
    }
)
nav.run()
