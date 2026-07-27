"""Streamlit frontend — single full-method showcase (Phase 4.2).

Shows, for a question: the answer, the cited spans HIGHLIGHTED in the source contract
text, the LIVE faithfulness score, and unsupported claims flagged in RED.

Architecture: the UI holds NO models and NO pipeline logic — it calls the FastAPI
backend, which loaded the embedder/reranker/NLI/indexes once at startup. Per the
Streamlit re-run gotcha, the HTTP client is created once via @st.cache_resource and
answers are memoized per question via @st.cache_data, so interaction never reloads
anything heavy.

NO baseline / tier / ablation comparison anywhere — this is the full method only.

Run:  streamlit run app/streamlit_app.py   (start the backend first: make api)
"""

from __future__ import annotations

import html

import httpx
import streamlit as st
import yaml

st.set_page_config(page_title="Citation-Grounded Legal QA", layout="wide")

HIGHLIGHT_COLOR = "#fde68a"  # supported citation (amber)
FLAG_COLOR = "#ef4444"       # unsupported claim (red)


@st.cache_resource
def get_config() -> dict:
    return yaml.safe_load(open("config/config.yaml", encoding="utf-8"))


@st.cache_resource
def get_client() -> httpx.Client:
    """Create the HTTP client ONCE (survives Streamlit re-runs)."""
    cfg = get_config()
    return httpx.Client(base_url=cfg["frontend"]["api_url"], timeout=120.0)


@st.cache_data(show_spinner=False)
def fetch_answer(question: str) -> dict:
    """Memoized POST /answer — identical questions don't re-hit the backend."""
    resp = get_client().post("/answer", json={"question": question})
    resp.raise_for_status()
    return resp.json()


@st.cache_data(show_spinner=False)
def fetch_contract(contract_id: str) -> str:
    resp = get_client().get(f"/contract/{contract_id}")
    resp.raise_for_status()
    return resp.json()["text"]


def _is_supported(claim: dict, threshold: float) -> bool:
    return (
        claim.get("entailment_label") == "entailment"
        and (claim.get("entailment_score") or 0.0) >= threshold
    )


def render_highlighted(text: str, ranges: list[tuple[int, int, str]]) -> str:
    """Render contract text as HTML with [start,end) ranges wrapped in colored marks.

    ``ranges`` is a list of (start, end, color); overlaps resolved by earliest start
    then widest span (later duplicates skipped).
    """
    ranges = sorted(ranges, key=lambda r: (r[0], -(r[1] - r[0])))
    out, cursor = [], 0
    for start, end, color in ranges:
        if start < cursor:  # skip overlaps already covered
            continue
        out.append(html.escape(text[cursor:start]))
        out.append(
            f'<mark style="background:{color};padding:0 2px;border-radius:3px">'
            f"{html.escape(text[start:end])}</mark>"
        )
        cursor = end
    out.append(html.escape(text[cursor:]))
    return "<div style='white-space:pre-wrap;font-family:monospace;font-size:0.85rem;"\
           "line-height:1.5'>" + "".join(out) + "</div>"


def main() -> None:
    cfg = get_config()
    threshold = cfg["verification"]["entailment_threshold"]

    st.title("⚖️ Citation-Grounded Legal QA")
    st.caption(
        "Every claim is checked against the source span it cites. "
        "Amber = supported citation · Red = unsupported (flagged) claim."
    )

    question = st.text_input(
        "Ask a question about the indexed contracts",
        placeholder="e.g. Can either party terminate for convenience, and is there a cap on liability?",
    )
    if not st.button("Answer", type="primary") or not question.strip():
        st.stop()

    try:
        with st.spinner("Retrieving, generating, and verifying …"):
            answer = fetch_answer(question.strip())
    except httpx.HTTPError as exc:
        st.error(f"Backend error: {exc}. Is the API running? (`make api`)")
        st.stop()

    claims = answer["claims"]
    flagged_texts = {c["text"] for c in answer["flagged_claims"]}
    score = answer["faithfulness_score"]

    # --- Headline: live faithfulness score ----------------------------------
    left, right = st.columns([1, 3])
    left.metric("Live faithfulness", f"{score:.0%}" if score is not None else "—")
    left.metric("Claims", len(claims))
    left.metric("Flagged (unsupported)", len(flagged_texts))
    right.markdown("### Answer")
    right.write(answer["raw_text"] or "_(no answer text)_")

    # --- Claims with per-claim verdicts -------------------------------------
    st.markdown("### Claims & citations")
    for c in claims:
        supported = _is_supported(c, threshold)
        badge = "✅ supported" if supported else "🚩 UNSUPPORTED"
        color = "green" if supported else "red"
        cit = c["citation"]
        st.markdown(
            f"- :{color}[{badge}] &nbsp; **{c['text']}**  \n"
            f"  ↳ `{cit['contract_id']}` chars {cit['char_start']}–{cit['char_end']} "
            f"· entailment={c.get('entailment_score') and round(c['entailment_score'], 3)}"
        )

    # --- Cited spans highlighted in the source contract(s) ------------------
    st.markdown("### Source with highlighted citations")
    contract_ids = sorted({c["citation"]["contract_id"] for c in claims})
    for cid in contract_ids:
        try:
            text = fetch_contract(cid)
        except httpx.HTTPError:
            st.warning(f"Could not load source text for {cid}")
            continue
        ranges = [
            (
                c["citation"]["char_start"],
                c["citation"]["char_end"],
                FLAG_COLOR if c["text"] in flagged_texts else HIGHLIGHT_COLOR,
            )
            for c in claims
            if c["citation"]["contract_id"] == cid
        ]
        with st.expander(f"📄 {cid}", expanded=True):
            st.markdown(render_highlighted(text, ranges), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
