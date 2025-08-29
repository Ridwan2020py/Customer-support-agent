"""
LangGraph Customer Support Agent — Full Python Skeleton

Implements an 11-stage customer support workflow with:
- Deterministic & non-deterministic stages
- State persistence across nodes
- MCP client orchestration (COMMON vs ATLAS)
- Structured logging and final payload output

This is a runnable skeleton with mock ability implementations.
Replace MCP client stubs with real integrations.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Literal, Optional, TypedDict
import asyncio
import logging
from copy import deepcopy

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

# -----------------------------
# Context & Config
# -----------------------------

class Context(TypedDict):
    """Runtime context for assistants / graph invocations.
    You can pass API keys, flags, tenant IDs, etc. via this context.
    """
    my_configurable_param: str
    solution_threshold: int  # e.g., 90

# Stage & ability registry mapping to MCP server routes
CONFIG: Dict[str, Any] = {
    "stages": {
        "INTAKE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "accept_payload", "client": "COMMON"},
            ],
        },
        "UNDERSTAND": {
            "mode": "deterministic",
            "abilities": [
                {"name": "parse_request_text", "client": "COMMON"},
                {"name": "extract_entities", "client": "ATLAS"},
            ],
        },
        "PREPARE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "normalize_fields", "client": "COMMON"},
                {"name": "enrich_records", "client": "ATLAS"},
                {"name": "add_flags_calculations", "client": "COMMON"},
            ],
        },
        "ASK": {
            "mode": "deterministic",
            "abilities": [
                {"name": "clarify_question", "client": "ATLAS"},
            ],
        },
        "WAIT": {
            "mode": "deterministic",
            "abilities": [
                {"name": "extract_answer", "client": "ATLAS"},
                {"name": "store_answer", "client": "COMMON"},  # state mgmt
            ],
        },
        "RETRIEVE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "knowledge_base_search", "client": "ATLAS"},
                {"name": "store_data", "client": "COMMON"},  # state mgmt
            ],
        },
        "DECIDE": {
            "mode": "non-deterministic",
            "abilities": [
                {"name": "solution_evaluation", "client": "COMMON"},
                {"name": "escalation_decision", "client": "ATLAS"},
                {"name": "update_payload", "client": "COMMON"},  # state mgmt
            ],
        },
        "UPDATE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "update_ticket", "client": "ATLAS"},
                {"name": "close_ticket", "client": "ATLAS"},
            ],
        },
        "CREATE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "response_generation", "client": "COMMON"},
            ],
        },
        "DO": {
            "mode": "deterministic",
            "abilities": [
                {"name": "execute_api_calls", "client": "ATLAS"},
                {"name": "trigger_notifications", "client": "ATLAS"},
            ],
        },
        "COMPLETE": {
            "mode": "deterministic",
            "abilities": [
                {"name": "output_payload", "client": "COMMON"},
            ],
        },
    },
}

# -----------------------------
# State Model
# -----------------------------

@dataclass
class State:
    # Input schema
    customer_name: str = ""
    email: str = ""
    query: str = ""
    priority: str = ""
    ticket_id: str = ""
    # Working memory
    parsed_request: Dict[str, Any] = field(default_factory=dict)
    entities: Dict[str, Any] = field(default_factory=dict)
    prepared: Dict[str, Any] = field(default_factory=dict)
    clarification_prompt: str = ""
    answer_from_user: str = ""
    kb_results: List[Dict[str, Any]] = field(default_factory=list)
    decision: Dict[str, Any] = field(default_factory=dict)  # {score, escalated}
    ticket_updates: Dict[str, Any] = field(default_factory=dict)
    reply_draft: str = ""
    actions_triggered: List[str] = field(default_factory=list)
    final_output: Dict[str, Any] = field(default_factory=dict)

# -----------------------------
# MCP Client Stubs (Replace these)
# -----------------------------

class CommonClient:
    """Mock COMMON MCP client for internal abilities (no external data)."""

    async def execute(self, ability: str, state: State, runtime: Runtime[Context]) -> Any:
        logging.info(f"[COMMON] Execute ability: {ability}")
        if ability == "accept_payload":
            return {"accepted": True}
        if ability == "parse_request_text":
            return {
                "text": state.query,
                "intent": "order_status" if "order" in state.query.lower() else "general",
                "parsed": True,
            }
        if ability == "normalize_fields":
            return {"priority": state.priority.upper(), "ticket_id": state.ticket_id.strip()}
        if ability == "add_flags_calculations":
            risk = "HIGH" if state.priority.lower() == "high" else "NORMAL"
            return {"sla_risk": risk}
        if ability == "store_answer":
            return {"stored": bool(state.answer_from_user)}
        if ability == "store_data":
            return {"kb_attached": len(state.kb_results)}
        if ability == "solution_evaluation":
            # naive heuristic: more KB hits → higher score
            base = 70 + min(30, 5 * len(state.kb_results))
            return min(100, base)
        if ability == "update_payload":
            return {"updated": True}
        if ability == "response_generation":
            who = state.customer_name or "Customer"
            if state.decision.get("escalated"):
                return f"Hi {who}, we've escalated your ticket {state.ticket_id} to a specialist. We'll update you shortly."
            else:
                return f"Hi {who}, we've resolved your issue. Ticket {state.ticket_id} is now closed."
        if ability == "output_payload":
            return deepcopy(asdict(state))
        # default no-op
        return {"noop": True}

class AtlasClient:
    """Mock ATLAS MCP client for external integrations (CRMs, KB, notifications)."""

    async def execute(self, ability: str, state: State, runtime: Runtime[Context]) -> Any:
        logging.info(f"[ATLAS] Execute ability: {ability}")
        if ability == "extract_entities":
            return {
                "product": "UnknownProduct" if "product" in state.query.lower() else None,
                "order_id": "ORD123" if "order" in state.query.lower() else None,
                "date": "2025-08-28",
            }
        if ability == "enrich_records":
            return {"sla_hours": 24 if state.priority.lower() == "high" else 72, "past_tickets": 2}
        if ability == "clarify_question":
            return "Could you share your order ID or any error message you see?"
        if ability == "extract_answer":
            # In a real system, this would wait for the human's reply. Here we read what's already in state.
            return state.answer_from_user or "N/A"
        if ability == "knowledge_base_search":
            # Pretend KB lookup using parsed intent/entities
            items = []
            if (state.parsed_request.get("intent") == "order_status") or state.entities.get("order_id"):
                items.append({"article_id": "KB-001", "title": "Track your order", "score": 0.92})
            else:
                items.append({"article_id": "KB-100", "title": "General troubleshooting", "score": 0.75})
            return items
        if ability == "escalation_decision":
            # Route to queue based on priority
            return {"assigned_queue": "L2-PRIORITY" if state.priority.lower() == "high" else "L2"}
        if ability == "update_ticket":
            return {"status": "in_progress", "last_updated": "2025-08-28T09:00:00Z"}
        if ability == "close_ticket":
            return {"status": "closed", "closed_at": "2025-08-28T09:05:00Z"}
        if ability == "execute_api_calls":
            return ["CRM:update_case", "OMS:check_order_status"]
        if ability == "trigger_notifications":
            return ["email:customer", "webhook:crm_activity"]
        # default no-op
        return {"noop": True}

COMMON = CommonClient()
ATLAS = AtlasClient()

# -----------------------------
# Utility: Router
# -----------------------------

async def route_ability(ability_name: str, client: Literal["COMMON", "ATLAS"], state: State, runtime: Runtime[Context]) -> Any:
    if client == "COMMON":
        return await COMMON.execute(ability_name, state, runtime)
    return await ATLAS.execute(ability_name, state, runtime)

# -----------------------------
# Stage Nodes
# -----------------------------

async def intake(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[INTAKE] start")
    await route_ability("accept_payload", "COMMON", state, runtime)
    logging.info("[INTAKE] accepted payload")
    return state

async def understand(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[UNDERSTAND] start")
    parsed = await route_ability("parse_request_text", "COMMON", state, runtime)
    state.parsed_request = parsed
    entities = await route_ability("extract_entities", "ATLAS", state, runtime)
    state.entities = entities
    logging.info(f"[UNDERSTAND] parsed={parsed} entities={entities}")
    return state

async def prepare(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[PREPARE] start")
    normalized = await route_ability("normalize_fields", "COMMON", state, runtime)
    enriched = await route_ability("enrich_records", "ATLAS", state, runtime)
    flags = await route_ability("add_flags_calculations", "COMMON", state, runtime)
    state.prepared = {**normalized, **enriched, **flags}
    logging.info(f"[PREPARE] prepared={state.prepared}")
    return state

async def ask(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[ASK] start")
    prompt = await route_ability("clarify_question", "ATLAS", state, runtime)
    state.clarification_prompt = prompt
    logging.info(f"[ASK] clarification_prompt='{prompt}'")
    return state

async def wait_for_answer(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[WAIT] start")
    answer = await route_ability("extract_answer", "ATLAS", state, runtime)
    state.answer_from_user = answer
    await route_ability("store_answer", "COMMON", state, runtime)
    logging.info(f"[WAIT] answer='{answer}'")
    return state

async def retrieve(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[RETRIEVE] start")
    kb = await route_ability("knowledge_base_search", "ATLAS", state, runtime)
    state.kb_results = kb
    await route_ability("store_data", "COMMON", state, runtime)
    logging.info(f"[RETRIEVE] kb_results={kb}")
    return state

async def decide(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[DECIDE] start")
    score = await route_ability("solution_evaluation", "COMMON", state, runtime)
    threshold = int(runtime.context.get("solution_threshold", 90))
    escalated = score < threshold
    details: Dict[str, Any] = {"score": int(score), "threshold": threshold, "escalated": escalated}
    if escalated:
        queue_info = await route_ability("escalation_decision", "ATLAS", state, runtime)
        details.update(queue_info)
    state.decision = details
    await route_ability("update_payload", "COMMON", state, runtime)
    logging.info(f"[DECIDE] decision={details}")
    return state

async def update(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[UPDATE] start")
    upd = await route_ability("update_ticket", "ATLAS", state, runtime)
    state.ticket_updates = {**state.ticket_updates, **upd}
    # Optionally close only if not escalated
    if not state.decision.get("escalated"):
        closed = await route_ability("close_ticket", "ATLAS", state, runtime)
        state.ticket_updates = {**state.ticket_updates, **closed}
    logging.info(f"[UPDATE] ticket_updates={state.ticket_updates}")
    return state

async def create(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[CREATE] start")
    reply = await route_ability("response_generation", "COMMON", state, runtime)
    state.reply_draft = str(reply)
    logging.info(f"[CREATE] reply_draft='{state.reply_draft}'")
    return state

async def do_actions(state: State, runtime: Runtime[Context]) -> State:
    logging.info("[DO] start")
    actions = await route_ability("execute_api_calls", "ATLAS", state, runtime)
    notifs = await route_ability("trigger_notifications", "ATLAS", state, runtime)
    state.actions_triggered = list(actions or []) + list(notifs or [])
    logging.info(f"[DO] actions_triggered={state.actions_triggered}")
    return state

async def complete(state: State, runtime: Runtime[Context]) -> Dict[str, Any]:
    logging.info("[COMPLETE] start")
    # Structured final payload
    payload = {
        "ticket_id": state.ticket_id,
        "customer": {"name": state.customer_name, "email": state.email},
        "priority": state.priority,
        "query": state.query,
        "parsed_request": state.parsed_request,
        "entities": state.entities,
        "prepared": state.prepared,
        "clarification_prompt": state.clarification_prompt,
        "answer_from_user": state.answer_from_user,
        "kb_results": state.kb_results,
        "decision": state.decision,
        "ticket_updates": state.ticket_updates,
        "reply_draft": state.reply_draft,
        "actions_triggered": state.actions_triggered,
    }
    state.final_output = payload
    logging.info("[COMPLETE] done")
    return payload

# -----------------------------
# Graph Wiring
# -----------------------------

graph = (
    StateGraph(State, context_schema=Context)
    .add_node(intake)
    .add_node(understand)
    .add_node(prepare)
    .add_node(ask)
    .add_node(wait_for_answer)
    .add_node(retrieve)
    .add_node(decide)
    .add_node(update)
    .add_node(create)
    .add_node(do_actions)
    .add_node(complete)
    # Edges (deterministic flow)
    .add_edge("__start__", "intake")
    .add_edge("intake", "understand")
    .add_edge("understand", "prepare")
    .add_edge("prepare", "ask")
    .add_edge("ask", "wait_for_answer")
    .add_edge("wait_for_answer", "retrieve")
    # Non-deterministic happens inside `decide`, but we continue flow regardless
    .add_edge("retrieve", "decide")
    .add_edge("decide", "update")
    .add_edge("update", "create")
    .add_edge("create", "do_actions")
    .add_edge("do_actions", "complete")
    .compile(name="Customer Support Workflow")
)

# -----------------------------
# Demo Runner (async)
# -----------------------------

async def demo_run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Initial input
    initial = State(
        customer_name="John Doe",
        email="john@example.com",
        query="My order hasn't arrived. Can you check the status?",
        priority="High",
        ticket_id="TCK-1001",
    )

    # Pretend a human later provided this answer (used by WAIT->extract_answer)
    initial.answer_from_user = "Order ID: ORD123"

    # Runtime context
    ctx: Context = {
        "my_configurable_param": "demo",
        "solution_threshold": 90,
    }

    # Execute the graph end-to-end
    # Note: Depending on LangGraph version, invocation may be graph.ainvoke(state_dict) or similar.
    # This skeleton expects LangGraph to pass Runtime[Context] automatically to nodes.
    result = await graph.ainvoke(initial, context=ctx)  # type: ignore[arg-type]

    # Print final structured payload
    print("\n===== FINAL PAYLOAD =====\n")
    print(result)

if __name__ == "__main__":
    try:
        asyncio.run(demo_run())
    except RuntimeError:
        # For environments where an event loop is already running (e.g., Jupyter)
        loop = asyncio.get_event_loop()
        loop.run_until_complete(demo_run())
