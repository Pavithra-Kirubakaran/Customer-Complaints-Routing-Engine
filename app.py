"""Enterprise Streamlit UI for the Complaint Routing Engine."""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import api_client

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Complaint Routing Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.25rem 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        min-height: 120px;
    }
    .kpi-label {
        font-size: 0.8rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value { font-size: 2rem; font-weight: 700; margin-top: 0.35rem; }
    .result-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .pipeline-step {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .success-banner {
        background: linear-gradient(135deg, #2e7d32, #43a047);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    div[data-testid="stSidebar"] { background: #f8fafc; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PRIORITY_COLORS = {
    "P1": "#d32f2f",
    "P2": "#f57c00",
    "P3": "#fbc02d",
    "P4": "#388e3c",
    "High": "#d32f2f",
    "Medium": "#f57c00",
    "Low": "#388e3c",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "page": "Dashboard",
        "last_result": None,
        "selected_ticket_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_supervisor():
    return api_client._direct_supervisor()


def load_data():
    api_client.ensure_db()
    tickets = api_client.fetch_tickets()
    summary = api_client.fetch_summary()
    escalations = api_client.fetch_escalations()
    guardrails = api_client.fetch_guardrails_summary()
    return tickets, summary, escalations, guardrails


def tickets_to_df(tickets: list[dict]) -> pd.DataFrame:
    if not tickets:
        return pd.DataFrame(
            columns=[
                "ticket_id", "customer_id", "channel", "subject", "message",
                "category", "priority", "queue", "team", "sla",
                "escalation_required", "assigned_at",
            ]
        )
    df = pd.DataFrame(tickets)
    df["assigned_at"] = pd.to_datetime(df["assigned_at"], errors="coerce")
    df["escalation_required"] = df["escalation_required"].astype(bool)
    if "hitl_required" not in df.columns:
        df["hitl_required"] = False
    else:
        df["hitl_required"] = df["hitl_required"].astype(bool)
    df["display_subject"] = df.apply(
        lambda r: r["subject"] or (str(r.get("message", ""))[:80] + "..."),
        axis=1,
    )
    return df


def kpi_card(label: str, value: str | int, icon: str = "📊"):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{icon} {label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


REASON_LABELS = {
    "urgent_routing": "Urgent routing rule",
    "high_priority": "High priority",
    "urgent_category": "Urgent category",
    "low_category_confidence": "Low category confidence",
    "low_priority_confidence": "Low priority confidence",
    "rag_no_confident_match": "RAG below similarity threshold",
    "output_normalized": "Output normalized by guardrails",
    "regulatory_keywords": "Regulatory / legal keywords",
    "high_value_claim": "High-value claim detected",
}


def format_reason(reason: str) -> str:
    return REASON_LABELS.get(reason, reason.replace("_", " ").title())


def priority_badge(priority: str) -> str:
    color = PRIORITY_COLORS.get(priority, "#64748b")
    return f"<span style='background:{color};color:white;padding:0.2rem 0.6rem;border-radius:6px;font-weight:600'>{priority}</span>"


def render_guardrail_badges(result: dict):
    badges = []
    if result.get("hitl_required"):
        badges.append("🧑‍💼 HITL review required")
    if result.get("escalation_required"):
        badges.append("🚨 Escalation flagged")
    if result.get("rag_guardrail_triggered"):
        badges.append("📚 RAG similarity guardrail triggered")
    for badge in badges:
        st.warning(badge)

    reasons = result.get("escalation_reasons") or []
    if reasons:
        st.markdown("**Escalation / HITL reasons**")
        st.write(", ".join(format_reason(reason) for reason in reasons))

    cat_conf = result.get("category_confidence")
    pri_conf = result.get("priority_confidence")
    if cat_conf is not None or pri_conf is not None:
        st.markdown("**Model confidence**")
        conf_bits = []
        if cat_conf is not None:
            conf_bits.append(f"Category: {cat_conf:.0%}")
        if pri_conf is not None:
            conf_bits.append(f"Priority: {pri_conf:.0%}")
        st.caption(" · ".join(conf_bits))

    if result.get("routing_explanation"):
        st.markdown("**Decision provenance**")
        st.code(result["routing_explanation"])


def render_rag_sources(result: dict):
    sources = result.get("rag_sources") or []
    if not sources:
        return
    st.markdown("**RAG sources**")
    for source in sources:
        status = "accepted" if source.get("accepted") else "filtered out"
        st.caption(
            f"{source.get('title', 'Unknown')} · id={source.get('id')} · "
            f"similarity={source.get('similarity')} · {status}"
        )
    if result.get("rag_kb_version"):
        st.caption(f"Knowledge base version: `{result['rag_kb_version']}`")


def render_audit_trace(ticket_id: int):
    try:
        audit = api_client.fetch_ticket_audit(int(ticket_id))
    except api_client.BackendError as exc:
        st.error(str(exc))
        return
    if not audit:
        st.info("No audit data available.")
        return

    with st.expander("Agent audit trace", expanded=False):
        trace = audit.get("audit_trace") or {}
        st.caption(
            f"{trace.get('step_count', 0)} steps · "
            f"{trace.get('total_duration_ms', 0)} ms total"
        )
        for step in trace.get("steps", []):
            st.markdown(f"**{step.get('node')}** — {step.get('duration_ms')} ms")
            st.json(step.get("output", {}))

        events = audit.get("events") or []
        if events:
            st.markdown("**Audit events**")
            for event in events:
                st.caption(f"{event.get('created_at')} · {event.get('event_type')}")
                st.json(event.get("payload", {}))


def render_routing_result(result: dict, show_audit: bool = False):
    cols = st.columns(4)
    cards = [
        ("Category", result.get("category", "—"), "🏷️"),
        ("Priority", result.get("priority", "—"), "⚡"),
        ("Queue", result.get("queue", "—"), "📥"),
        ("Team", result.get("team", "—"), "👥"),
    ]
    for col, (label, value, icon) in zip(cols, cards):
        with col:
            st.markdown(
                f"<div class='result-card'><strong  style='color: #3B82F6;'>{icon} {label}</strong><br><span style='font-size:1.25rem;color: #84ADF2;'>{value}</span></div>",
                unsafe_allow_html=True,
            )

    render_guardrail_badges(result)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**SLA target**")
        st.info(result.get("sla", "—"))
        st.markdown("**Routing note**")
        st.write(result.get("routing_note") or "—")
        if result.get("context"):
            st.markdown("**AI context summary**")
            st.write(result["context"])
    with col2:
        st.markdown("**Knowledge base (RAG)**")
        st.write(result.get("rag_context") or "No related articles found.")
        render_rag_sources(result)
        if result.get("monitoring_note"):
            st.markdown("**Monitoring note**")
            st.write(result["monitoring_note"])

    if show_audit and result.get("ticket_id"):
        render_audit_trace(result["ticket_id"])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_dashboard(summary: dict, df: pd.DataFrame, escalations: list, guardrails: dict):
    st.title("📊 Dashboard")
    st.caption("Live overview from your routing engine and agent pipeline.")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi_card("Total tickets", summary.get("total_tickets", 0), "📋")
    with c2:
        kpi_card("Escalations", summary.get("escalation_count", 0), "🚨")
    with c3:
        kpi_card("HITL reviews", summary.get("hitl_count", 0), "🧑‍💼")
    with c4:
        kpi_card("RAG guardrails", summary.get("rag_guardrail_count", 0), "📚")
    with c5:
        high = sum(
            summary.get("by_priority", {}).get(p, 0)
            for p in ("P1", "P2", "High", "Critical")
        )
        kpi_card("High priority", high, "⚡")

    st.markdown("---")

    if df.empty:
        st.info("No tickets yet. Submit your first complaint from **Submit Ticket**.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tickets by category")
        by_cat = summary.get("by_category") or df["category"].value_counts().to_dict()
        if by_cat:
            cat_df = pd.DataFrame({"category": list(by_cat.keys()), "count": list(by_cat.values())})
            fig = px.bar(
                cat_df.sort_values("count", ascending=True),
                x="count", y="category", orientation="h",
                color="count", color_continuous_scale="Blues",
            )
            fig.update_layout(height=360, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Priority distribution")
        by_pri = summary.get("by_priority") or df["priority"].value_counts().to_dict()
        if by_pri:
            colors = [PRIORITY_COLORS.get(p, "#667eea") for p in by_pri.keys()]
            fig = go.Figure(data=[go.Pie(labels=list(by_pri.keys()), values=list(by_pri.values()), marker=dict(colors=colors))])
            fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Volume over time")
        daily = df.groupby(df["assigned_at"].dt.date).size().reset_index(name="tickets")
        daily.columns = ["date", "tickets"]
        fig = px.line(daily, x="date", y="tickets", markers=True)
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Recent escalations")
        if escalations:
            esc_df = pd.DataFrame(escalations)[
                ["ticket_id", "priority", "team", "category", "hitl_required", "assigned_at"]
            ].head(8)
            st.dataframe(esc_df, use_container_width=True, hide_index=True)
        else:
            st.success("No pending escalations.")


def page_submit():
    st.title("📝 Submit a complaint")
    st.markdown(
        """
        <div style='background:#eff6ff;color: #3B82F6;padding:1rem;border-radius:8px;border-left:4px solid #3b82f6;margin-bottom:1rem'>
        Describe your issue below. Our AI agents will classify, prioritize, and route it to the right team automatically.
        Do not include credit card numbers or Social Security numbers. Email and phone numbers are redacted automatically.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("ticket_form", clear_on_submit=False):
        col1, col2 = st.columns(2)
        with col1:
            customer_id = st.text_input("Customer ID *", placeholder="cust-001 or email@company.com")
            channel = st.selectbox("Channel *", ["web", "email", "chat", "phone"])
        with col2:
            subject = st.text_input("Subject", placeholder="Brief summary of the issue")
            st.caption("Optional — helps agents route faster")

        message = st.text_area(
            "Complaint details *",
            height=160,
            placeholder="Describe your issue in detail (minimum 20 characters)...",
        )
        submitted = st.form_submit_button("🤖 Analyze & route ticket", type="primary", use_container_width=True)

    if not submitted:
        if st.session_state.last_result:
            st.markdown("---")
            st.subheader("Last submission")
            render_routing_result(st.session_state.last_result)
        return

    errors = []
    if not customer_id.strip():
        errors.append("Customer ID is required.")
    if len(message.strip()) < 20:
        errors.append("Please enter at least 20 characters in the complaint details.")
    if errors:
        for err in errors:
            st.error(err)
        return

    payload = {
        "customer_id": customer_id.strip(),
        "channel": channel,
        "subject": subject.strip() or None,
        "message": message.strip(),
    }

    with st.spinner("Running AI agents — classification, RAG lookup, routing..."):
        try:
            supervisor = None if api_client._use_api() else get_supervisor()
            result = api_client.submit_ticket(payload, supervisor=supervisor)
            st.session_state.last_result = result
            st.session_state.selected_ticket_id = result.get("ticket_id")
            st.cache_data.clear()
        except api_client.BackendError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Routing failed: {exc}")
            return

    st.markdown(
        f"""
        <div class="success-banner">
            <h3 style="margin:0 0 0.5rem 0">✅ Ticket #{result['ticket_id']} created</h3>
            <p style="margin:0;opacity:0.95">Routed to <strong>{result.get('team')}</strong> · Priority <strong>{result.get('priority')}</strong> · SLA <strong>{result.get('sla')}</strong></p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_routing_result(result)


def page_explorer(df: pd.DataFrame):
    st.title("🔍 Ticket explorer")
    if df.empty:
        st.info("No tickets to explore yet.")
        return

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        search = st.text_input("Search", placeholder="Subject or message...")
    with col2:
        categories = st.multiselect("Category", sorted(df["category"].dropna().unique()))
    with col3:
        priorities = st.multiselect("Priority", sorted(df["priority"].dropna().unique()))
    with col4:
        teams = st.multiselect("Team", sorted(df["team"].dropna().unique()))

    filtered = df.copy()
    if search:
        mask = (
            filtered["display_subject"].str.contains(search, case=False, na=False)
            | filtered["message"].astype(str).str.contains(search, case=False, na=False)
            | filtered["customer_id"].astype(str).str.contains(search, case=False, na=False)
        )
        filtered = filtered[mask]
    if categories:
        filtered = filtered[filtered["category"].isin(categories)]
    if priorities:
        filtered = filtered[filtered["priority"].isin(priorities)]
    if teams:
        filtered = filtered[filtered["team"].isin(teams)]

    st.caption(f"Showing {len(filtered)} of {len(df)} tickets")
    display = filtered[
        ["ticket_id", "customer_id", "channel", "display_subject", "category", "priority", "team", "sla", "escalation_required", "hitl_required", "assigned_at"]
    ].rename(columns={"display_subject": "subject"})
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("View ticket details")
    ticket_ids = filtered["ticket_id"].tolist()
    if ticket_ids:
        selected = st.selectbox(
            "Select ticket",
            ticket_ids,
            format_func=lambda x: f"#{x} — {filtered.loc[filtered['ticket_id']==x, 'display_subject'].iloc[0]}",
            index=0,
        )
        if st.button("Open ticket details"):
            st.session_state.selected_ticket_id = selected
            st.session_state.page = "Ticket Details"
            st.rerun()


def page_ticket_details(df: pd.DataFrame):
    st.title("🤖 Ticket details & AI routing")
    if df.empty:
        st.info("No tickets available.")
        return

    ticket_ids = df["ticket_id"].tolist()
    default_idx = 0
    if st.session_state.selected_ticket_id in ticket_ids:
        default_idx = ticket_ids.index(st.session_state.selected_ticket_id)

    selected_id = st.selectbox(
        "Ticket",
        ticket_ids,
        index=default_idx,
        format_func=lambda x: f"#{x} — {df.loc[df['ticket_id']==x, 'display_subject'].iloc[0]}",
    )
    st.session_state.selected_ticket_id = selected_id

    try:
        ticket = api_client.fetch_ticket(int(selected_id))
    except api_client.BackendError as exc:
        st.error(str(exc))
        return

    if not ticket:
        st.warning("Ticket not found.")
        return

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Complaint")
        st.write(f"**Customer:** {ticket.get('customer_id')}")
        st.write(f"**Channel:** {ticket.get('channel')}")
        st.write(f"**Submitted:** {ticket.get('assigned_at', '—')}")
        if ticket.get("subject"):
            st.write(f"**Subject:** {ticket['subject']}")
        st.markdown("---")
        st.write(ticket.get("message") or "—")

    with col2:
        st.subheader("AI routing output")
        render_routing_result(ticket, show_audit=True)

        pri = ticket.get("priority", "")
        score_map = {"P1": 95, "P2": 75, "P3": 50, "P4": 25, "High": 85, "Medium": 55, "Low": 30}
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score_map.get(pri, 50),
            title={"text": "Priority level"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": PRIORITY_COLORS.get(pri, "#667eea")},
                "steps": [
                    {"range": [0, 25], "color": "#e8f5e9"},
                    {"range": [25, 50], "color": "#fff9c4"},
                    {"range": [50, 75], "color": "#ffe0b2"},
                    {"range": [75, 100], "color": "#ffcdd2"},
                ],
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)


def page_escalations(escalations: list):
    st.title("🚨 Escalation & HITL monitor")
    st.caption("Tickets flagged for supervisor review or human-in-the-loop handling.")

    if not escalations:
        st.success("No pending escalations — all tickets are within normal handling.")
        return

    hitl_count = sum(1 for ticket in escalations if ticket.get("hitl_required"))
    st.warning(f"{len(escalations)} ticket(s) flagged · {hitl_count} require HITL review")
    esc_df = pd.DataFrame(escalations)
    cols = [
        "ticket_id", "customer_id", "category", "priority", "team", "queue",
        "sla", "hitl_required", "escalation_reasons", "routing_explanation", "assigned_at",
    ]
    available = [c for c in cols if c in esc_df.columns]
    st.dataframe(esc_df[available], use_container_width=True, hide_index=True)

    for ticket in escalations[:5]:
        reasons = ticket.get("escalation_reasons") or []
        reason_text = ", ".join(format_reason(r) for r in reasons) if reasons else "No reason codes"
        with st.expander(
            f"Ticket #{ticket['ticket_id']} — {ticket.get('category')} / {ticket.get('priority')} · {reason_text}"
        ):
            st.write(ticket.get("routing_note") or "—")
            if ticket.get("routing_explanation"):
                st.code(ticket["routing_explanation"])
            if st.button("View full details", key=f"esc_{ticket['ticket_id']}"):
                st.session_state.selected_ticket_id = ticket["ticket_id"]
                st.session_state.page = "Ticket Details"
                st.rerun()





def page_guardrails(guardrails: dict, summary: dict):
    st.title("🛡️ Guardrails & audit")
    st.caption("Operational guardrail metrics across routing, RAG, HITL, and output validation.")

    if guardrails.get("total_tickets", 0) == 0:
        st.info("Submit tickets to populate guardrail analytics.")
        return

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("HITL rate", f"{guardrails.get('hitl_rate', 0) * 100:.1f}%", "🧑‍💼")
    with c2:
        kpi_card("Escalation rate", f"{guardrails.get('escalation_rate', 0) * 100:.1f}%", "🚨")
    with c3:
        kpi_card("RAG guardrail rate", f"{guardrails.get('rag_guardrail_rate', 0) * 100:.1f}%", "📚")
    with c4:
        kpi_card("Output corrections", summary.get("output_correction_count", 0), "✅")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Average model confidence")
        st.metric("Category", guardrails.get("avg_category_confidence") or "—")
        st.metric("Priority", guardrails.get("avg_priority_confidence") or "—")
    with col2:
        st.subheader("Top escalation / HITL reasons")
        reasons = guardrails.get("top_escalation_reasons") or {}
        if reasons:
            reason_df = pd.DataFrame(
                {"reason": [format_reason(k) for k in reasons.keys()], "count": list(reasons.values())}
            )
            fig = px.bar(reason_df, x="count", y="reason", orientation="h")
            fig.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No escalation reason codes recorded yet.")


def page_analytics(summary: dict, df: pd.DataFrame, guardrails: dict):
    st.title("📈 Analytics")
    if df.empty:
        st.info("Submit tickets to see analytics.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tickets by team")
        by_team = summary.get("by_team") or df["team"].value_counts().to_dict()
        team_df = pd.DataFrame({"team": list(by_team.keys()), "count": list(by_team.values())})
        fig = px.bar(team_df.sort_values("count", ascending=True), x="count", y="team", orientation="h", color="count", color_continuous_scale="Viridis")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Channel breakdown")
        channel_counts = df["channel"].value_counts()
        fig = px.pie(names=channel_counts.index, values=channel_counts.values)
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Escalation rate by category")
    if "escalation_required" in df.columns:
        esc_rate = df.groupby("category")["escalation_required"].mean().reset_index()
        esc_rate.columns = ["category", "escalation_rate"]
        esc_rate["escalation_rate"] = (esc_rate["escalation_rate"] * 100).round(1)
        fig = px.bar(esc_rate.sort_values("escalation_rate"), x="escalation_rate", y="category", orientation="h", labels={"escalation_rate": "Escalation %"})
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)


def page_settings():
    st.title("⚙️ Settings")
    st.subheader("Backend connection")
    mode = st.radio(
        "Connection mode",
        ["Direct (same process — recommended)", "HTTP API (requires uvicorn)"],
        index=0 if not api_client._use_api() else 1,
    )
    api_url = st.text_input("API URL", value=api_client._api_url())

    st.info(
        "Direct mode runs agents in-process — start with `streamlit run app.py` only. "
        "API mode requires `uvicorn main:app --reload` in a separate terminal."
    )

    if st.button("Apply & refresh"):
        os.environ["USE_API"] = "true" if "HTTP" in mode else "false"
        os.environ["API_URL"] = api_url.strip().rstrip("/")
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Settings applied. Reload the page if connection mode changed.")
        st.rerun()

    st.markdown("---")
    st.subheader("Environment")
    st.write(f"- **USE_API:** `{os.getenv('USE_API', 'false')}`")
    st.write(f"- **API_URL:** `{api_client._api_url()}`")
    st.write(f"- **GOOGLE_API_KEY:** {'set ✓' if os.getenv('GOOGLE_API_KEY') else 'not set (dummy LLM used)'}")

    if st.button("🔄 Refresh all data"):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def cached_load_data():
    return load_data()


def main():
    init_state()

    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center;padding:0.5rem 0 1rem'>
                <div style='font-size:2rem'>🎯</div>
                <h2 style='margin:0;color:#1e40af;font-size:1.1rem'>Routing Engine</h2>
                <p style='color:#64748b;font-size:0.8rem;margin:0'>AI complaint classification</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = [
            ("📊", "Dashboard"),
            ("📝", "Submit Ticket"),
            ("🔍", "Ticket Explorer"),
            ("🤖", "Ticket Details"),
            ("🚨", "Escalations"),
            ("🛡️", "Guardrails & Audit"),
            ("📈", "Analytics"),
            ("⚙️", "Settings"),
        ]
        for icon, name in pages:
            if st.button(f"{icon}  {name}", use_container_width=True, key=f"nav_{name}"):
                st.session_state.page = name
                st.rerun()

        st.markdown("---")
        mode_label = "API" if api_client._use_api() else "Direct"
        st.caption(f"Backend: **{mode_label}** · {datetime.now().strftime('%H:%M')}")

        if st.button("↻ Refresh data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    try:
        tickets, summary, escalations, guardrails = cached_load_data()
    except api_client.BackendError as exc:
        st.error(str(exc))
        st.stop()

    df = tickets_to_df(tickets)
    page = st.session_state.page

    if page == "Dashboard":
        page_dashboard(summary, df, escalations, guardrails)
    elif page == "Submit Ticket":
        page_submit()
    elif page == "Ticket Explorer":
        page_explorer(df)
    elif page == "Ticket Details":
        page_ticket_details(df)
    elif page == "Escalations":
        page_escalations(escalations)
    elif page == "Guardrails & Audit":
        page_guardrails(guardrails, summary)
    elif page == "Analytics":
        page_analytics(summary, df, guardrails)
    elif page == "Settings":
        page_settings()
    else:
        page_dashboard(summary, df, escalations, guardrails)


if __name__ == "__main__":
    main()
