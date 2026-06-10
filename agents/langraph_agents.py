import os
from types import SimpleNamespace
from dotenv import load_dotenv, dotenv_values
from datetime import datetime
from typing import Callable
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage  # used in _generate_text
from langgraph.graph import StateGraph, END

from agents.state import AgentState
from agents.tools import (
    classify_category,
    predict_priority,
    search_knowledge_base,
    determine_sla,
    route_ticket,
    check_escalation,
)


def _call_tool(tool_obj, *args, **kwargs):
    """Call a tool which may be a LangChain StructuredTool or a plain function.

    Supports StructuredTool.run, StructuredTool.func, or direct callable.
    """
    if hasattr(tool_obj, "func") and callable(getattr(tool_obj, "func")):
        return tool_obj.func(*args, **kwargs)
    if hasattr(tool_obj, "run") and callable(getattr(tool_obj, "run")):
        return tool_obj.run(*args, **kwargs)
    if callable(tool_obj):
        return tool_obj(*args, **kwargs)
    raise RuntimeError(f"Tool {tool_obj} is not callable")


def _init_llm():
    """Initialize Google Generative AI LLM."""
    # Load .env if present (does not override existing environment variables)
    load_dotenv()

    env_values = dotenv_values()
    api_key = env_values.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        # If there's no Google API key, return a dummy LLM so the workflow can be
        # exercised locally without failing startup. The dummy LLM implements a
        # minimal `generate()` that returns an object compatible with
        # `_generate_text`'s expectations.
        print("Warning: GOOGLE_API_KEY not set — using dummy LLM for local testing.")

        class DummyLLM:
            def generate(self, messages):
                # messages is a nested list; we just return a canned response
                return SimpleNamespace(generations=[[SimpleNamespace(text="ok")]])

        return DummyLLM()

    MODEL_NAME = "gemini-2.5-flash"
    print(f"Initializing Google model: {MODEL_NAME}")
    try:
        llm_instance = ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=api_key,
            temperature=0.0,
        )
        return llm_instance
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize Google model '{MODEL_NAME}': {exc}"
        ) from exc


# Lazy/robust LLM init: attempt to initialize the Google LLM, but fall back to
# a local dummy LLM for testing when the environment key is not present or
# when model negotiation fails. This lets us run end-to-end tests without a
# live Google API key.
try:
    llm = _init_llm()
except Exception as _exc:
    print(f"Warning: Google LLM not initialized: {_exc}. Using local dummy LLM for testing.")

    from types import SimpleNamespace

    class _DummyLLM:
        def generate(self, messages):
            # Very small deterministic response for testing flows.
            text = "This is a dummy LLM response for testing."
            return SimpleNamespace(generations=[[SimpleNamespace(text=text)]])

    llm = _DummyLLM()


def _generate_text(prompt: str) -> str:
    """Invoke the chat LLM with a prompt using the Google chat interface."""
    if not hasattr(llm, "generate"):
        raise RuntimeError("ChatGoogleGenerativeAI requires the generate() method.")

    system_message = SystemMessage(content="You are a helpful complaint routing assistant.")
    human_message = HumanMessage(content=prompt)
    response = llm.generate([[system_message, human_message]])

    if hasattr(response, "generations") and response.generations:
        first_batch = response.generations[0]
        if first_batch and hasattr(first_batch[0], "text"):
            return first_batch[0].text

    # Fallback for other response formats
    if hasattr(response, "text"):
        return str(response.text)
    if isinstance(response, str):
        return response

    raise RuntimeError("Unable to parse response from the LLM.")


def supervisor_agent(state: AgentState) -> AgentState:
    """Entry agent that prepares the complaint state for the LangGraph workflow."""
    complaint_text = state["message"]
    if state.get("subject"):
        complaint_text = f"Subject: {state['subject']}\n\n{complaint_text}"

    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the supervisor agent for complaint routing. "
                    "Your job is to coordinate the downstream agents and maintain state."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Process this customer complaint for routing.\n\n{complaint_text}\n\n"
                    f"Channel: {state['channel']}\nCustomer ID: {state['customer_id']}"
                )
            },
        ]
    }


def context_agent(state: AgentState) -> AgentState:
    """Create the complaint context for downstream agents."""
    prompt = (
        "Summarize this customer complaint into a concise context of what the issue is, "
        "including any product, billing, support, or outage details. "
        "Return a short paragraph.\n\n"
        f"Complaint:\n{state['message']}"
    )

    return {"context": _generate_text(prompt).strip()}


def category_agent(state: AgentState) -> AgentState:
    """Classify the complaint category using the trained model."""
    text = state.get("context") or state["message"]
    return {"category": _call_tool(classify_category, text)}


def priority_agent(state: AgentState) -> AgentState:
    """Predict the complaint priority using the trained model."""
    text = state.get("context") or state["message"]
    return {"priority": _call_tool(predict_priority, text)}


def rag_agent(state: AgentState) -> AgentState:
    """Retrieve relevant support knowledge based on complaint context."""
    query_text = state.get("context") or state["message"]
    return {"rag_context": _call_tool(search_knowledge_base, query_text)}


def sla_agent(state: AgentState) -> AgentState:
    """Determine SLA requirements from priority and category."""
    # Be defensive: category or priority may be missing earlier in the flow.
    priority = state.get("priority") or "low"
    category = state.get("category") or "General Inquiry"
    return {"sla": _call_tool(determine_sla, priority, category)}


def routing_agent(state: AgentState) -> AgentState:
    """Assign the ticket to queue and team and create a routing note."""
    category = state.get("category") or "General Inquiry"
    priority = state.get("priority") or "low"
    channel = state.get("channel") or "email"
    route_result = _call_tool(route_ticket, category, priority, channel)
    return {
        "queue": route_result["queue"],
        "team": route_result["team"],
        "routing_note": route_result["routing_note"],
    }


def escalation_agent(state: AgentState) -> AgentState:
    """Decide whether the ticket needs escalation."""
    priority = state.get("priority") or "low"
    category = state.get("category") or "General Inquiry"
    return {"escalation_required": _call_tool(check_escalation, priority, category)}


def monitoring_agent(state: AgentState) -> AgentState:
    """Create monitoring metadata for the ticket."""
    return {
        "monitoring_note": (
            f"Ticket routed to {state['team']} with priority {state['priority']}. "
            f"Escalation required: {state['escalation_required']}"
        )
    }


def finalize_agent(state: AgentState) -> AgentState:
    """Finalize any missing ticket fields before persistence."""
    category = state.get("category") or "General Inquiry"
    return {
        "category": category,
        "priority": state.get("priority") or "low",
        "queue": state.get("queue") or category,
        "team": state.get("team") or "Customer Success Team",
        "sla": state.get("sla") or "72 Hours",
        "routing_note": state.get("routing_note") or "Standard routing applied.",
        "rag_context": state.get("rag_context") or "No relevant context retrieved.",
        "escalation_required": bool(state.get("escalation_required")),
    }


def create_routing_graph() -> Callable[[AgentState], AgentState]:
    """Create the LangGraph state machine with parallel fan-out for independent agents.

    Topology
    --------
    supervisor
        └─► context_agent
                ├─► category_agent ─┐
                ├─► priority_agent  ├─► sla_agent ─► routing_agent ─► escalation_agent
                └─► rag_agent      ─┘                                        │
                                                                     monitoring_agent
                                                                             │
                                                                     finalize_agent ─► END

    After context_agent, three agents run **in parallel**:
    - category_agent  (classifies complaint type)
    - priority_agent  (predicts urgency level)
    - rag_agent       (retrieves knowledge-base context)

    sla_agent acts as the **join barrier** — LangGraph 1.x waits for all three
    incoming branches to complete before executing sla_agent.
    """
    workflow = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    workflow.add_node("supervisor", supervisor_agent)
    workflow.add_node("context_agent", context_agent)
    workflow.add_node("category_agent", category_agent)
    workflow.add_node("priority_agent", priority_agent)
    workflow.add_node("rag_agent", rag_agent)
    workflow.add_node("sla_agent", sla_agent)
    workflow.add_node("routing_agent", routing_agent)
    workflow.add_node("escalation_agent", escalation_agent)
    workflow.add_node("monitoring_agent", monitoring_agent)
    workflow.add_node("finalize_agent", finalize_agent)

    # ── Sequential preamble ───────────────────────────────────────────────────
    workflow.set_entry_point("supervisor")
    workflow.add_edge("supervisor", "context_agent")

    # ── Fan-out: 3 independent agents run in parallel ─────────────────────────
    workflow.add_edge("context_agent", "category_agent")
    workflow.add_edge("context_agent", "priority_agent")
    workflow.add_edge("context_agent", "rag_agent")

    # ── Fan-in: sla_agent is the barrier — starts only after all 3 finish ─────
    workflow.add_edge("category_agent", "sla_agent")
    workflow.add_edge("priority_agent", "sla_agent")
    workflow.add_edge("rag_agent", "sla_agent")

    # ── Sequential tail ───────────────────────────────────────────────────────
    workflow.add_edge("sla_agent", "routing_agent")
    workflow.add_edge("routing_agent", "escalation_agent")
    workflow.add_edge("escalation_agent", "monitoring_agent")
    workflow.add_edge("monitoring_agent", "finalize_agent")
    workflow.add_edge("finalize_agent", END)

    return workflow.compile()



routing_graph = create_routing_graph()
