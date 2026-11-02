"""LangGraph implementation of the proposed Agentic RAG lead-qualification agent.

Ablation switches:
  sticky    -- once lead collection starts, every turn goes to the extractor regardless of the classified intent.
  interrupt -- during lead collection, a product question is answered by the RAG node, which then resumes collection.
  confirm   -- before the CRM write, the collected details are read back and must be explicitly confirmed.
Tool execution is always structurally gated: tool_node is reachable only when all lead fields are present
(and, with confirm, only on the turn the user affirms them).
"""
import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from . import config, llm, prompts
from .extractor import extract_fields, is_affirmative, is_decline, is_negative
from .retriever import Retriever, format_context


class AgentState(TypedDict, total=False):
    user_input: str
    history: list                      # previous turns as chat messages
    intent: Optional[str]
    response: str
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    awaiting: Optional[str]            # lead field the assistant last asked for
    collecting_lead: bool
    lead_collected: bool
    declined: bool
    confirming: bool                   # details read back, waiting for the user's confirmation
    affirmed: bool                     # this turn confirms the read-back details
    negated: bool                      # this turn rejects them without giving a new value
    new_fields: list                   # lead fields captured this turn
    crm_records: list                  # backend writes made by tool_node
    retrieved: list                    # chunk ids retrieved this turn
    query: Optional[str]               # standalone retrieval query this turn
    trace: Annotated[list, operator.add]


def new_state() -> AgentState:
    return {"history": [], "collecting_lead": False, "lead_collected": False, "confirming": False,
            "crm_records": [], "name": None, "email": None, "platform": None, "awaiting": None}


def parse_intent(raw: str) -> tuple[str, bool]:
    """Map raw classifier output to a label; returns (label, parsed_ok)."""
    t = raw.lower().replace(" ", "_")
    found = [(t.find(l), l) for l in config.INTENTS if l in t]
    if not found:
        return "general", False
    return min(found)[1], True


def classify_intent(text: str, prev_assistant: str = "") -> tuple[str, bool]:
    raw = llm.chat([
        {"role": "system", "content": prompts.INTENT_SYSTEM},
        {"role": "user", "content": prompts.INTENT_USER.format(prev=prev_assistant or "(none)", text=text)},
    ], node="intent", max_tokens=8)
    return parse_intent(raw)


def _last_assistant(history: list) -> str:
    return next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")


def _missing(s: AgentState) -> list[str]:
    return [f for f in config.LEAD_FIELDS if not s.get(f)]


def _lead_question(field: str, s: AgentState) -> str:
    q = prompts.LEAD_QUESTIONS[field]
    return q.format(name_part=f", {s['name']}" if s.get("name") else "") if field == "email" else q


def _is_question(s: AgentState) -> bool:
    """A product question asked during lead collection (a turn that only supplied a field is not one)."""
    if s.get("intent") == "high_intent":
        return False  # "how do I sign up?" is answered by the lead flow itself, not by retrieval
    if "?" in s["user_input"]:
        return True
    return s.get("intent") == "product_inquiry" and not s.get("new_fields")


def _confirm_text(s: AgentState) -> str:
    return prompts.CONFIRM_ASK.format(name=s["name"], email=s["email"], platform=s["platform"])


def build_graph(retriever: Retriever, sticky: bool = True, interrupt: bool = True, confirm: bool = True):

    def intent_node(s: AgentState):
        label, _ = classify_intent(s["user_input"], _last_assistant(s.get("history", [])))
        return {"intent": label, "trace": ["intent"]}

    def route_intent(s: AgentState) -> str:
        if sticky and s.get("collecting_lead") and not s.get("lead_collected"):
            return "extract"
        if s["intent"] == "high_intent" and not s.get("lead_collected"):
            return "extract"
        return {"greeting": "greeting", "product_inquiry": "rag"}.get(s["intent"], "general")

    def greeting_node(s: AgentState):
        return {"response": prompts.GREETING_TEXT, "trace": ["greeting"]}

    def general_node(s: AgentState):
        if s.get("intent") == "high_intent" and s.get("lead_collected"):
            return {"response": prompts.ALREADY_REGISTERED_TEXT, "trace": ["general"]}
        reply = llm.chat([{"role": "system", "content": prompts.GENERAL_SYSTEM}, *s.get("history", [])[-4:],
                          {"role": "user", "content": s["user_input"]}], node="general")
        return {"response": reply, "trace": ["general"]}

    def rag_node(s: AgentState):
        history = s.get("history", [])
        query = s["user_input"]
        if history:
            query = llm.chat([{"role": "system", "content": prompts.CONDENSE_SYSTEM}, *history[-6:],
                              {"role": "user", "content": s["user_input"]}], node="condense", max_tokens=64)
        hits = retriever.search(query)
        answer = llm.chat([
            {"role": "system", "content": prompts.RAG_SYSTEM.format(context=format_context(hits))},
            *history[-6:], {"role": "user", "content": s["user_input"]},
        ], node="rag")
        upd = {"query": query, "retrieved": [h["chunk_id"] for h in hits], "trace": ["rag"]}
        if s.get("collecting_lead") and not s.get("lead_collected"):
            if _missing(s):
                field = _missing(s)[0]
                answer += prompts.RESUME_PREFIX + _lead_question(field, s)
                upd["awaiting"] = field
            elif confirm:
                answer += prompts.RESUME_PREFIX + _confirm_text(s)
                upd["confirming"] = True
        upd["response"] = answer
        return upd

    def extract_node(s: AgentState):
        found = extract_fields(s["user_input"], s.get("awaiting"))
        upd = {"collecting_lead": True, "trace": ["extract"]}
        if found["name"] and (not s.get("name") or s.get("awaiting") == "name"):
            upd["name"] = found["name"]
        if found["email"]:
            upd["email"] = found["email"]          # later emails overwrite earlier ones (corrections)
        if found["platform"]:
            upd["platform"] = found["platform"]
        upd["new_fields"] = [f for f in config.LEAD_FIELDS if f in upd]
        if is_decline(s["user_input"]) and not any(found.values()):
            upd["declined"] = True
        elif s.get("confirming") and not upd["new_fields"]:
            upd["affirmed"] = is_affirmative(s["user_input"])
            upd["negated"] = is_negative(s["user_input"])
        return upd

    def route_after_extract(s: AgentState) -> str:
        if s.get("declined"):
            return "decline"
        if _missing(s):
            return "rag" if interrupt and _is_question(s) else "lead"
        if confirm and not s.get("affirmed"):
            if interrupt and _is_question(s) and not s.get("new_fields"):
                return "rag"                     # answer, then re-ask for confirmation
            return "confirm"
        return "tool"

    def confirm_node(s: AgentState):
        text = prompts.CONFIRM_FIX_TEXT if s.get("negated") else _confirm_text(s)
        return {"response": text, "confirming": True, "awaiting": None, "trace": ["confirm"]}

    def lead_node(s: AgentState):
        field = _missing(s)[0]
        return {"response": _lead_question(field, s), "awaiting": field, "trace": ["lead"]}

    def tool_node(s: AgentState):
        assert not _missing(s), "tool_node reached with missing fields"
        record = {"name": s["name"], "email": s["email"], "platform": s["platform"], "ts": time.time()}
        return {
            "crm_records": s.get("crm_records", []) + [record],
            "lead_collected": True, "collecting_lead": False, "confirming": False, "awaiting": None,
            "response": prompts.CONFIRM_TEXT.format(**record),
            "trace": ["tool"],
        }

    def decline_node(s: AgentState):
        return {"response": prompts.DECLINE_TEXT, "collecting_lead": False, "confirming": False,
                "awaiting": None, "trace": ["decline"]}

    g = StateGraph(AgentState)
    for name, fn in [("intent", intent_node), ("greeting", greeting_node), ("general", general_node),
                     ("rag", rag_node), ("extract", extract_node), ("lead", lead_node), ("confirm", confirm_node),
                     ("tool", tool_node), ("decline", decline_node)]:
        g.add_node(name, fn)
    g.add_edge(START, "intent")
    g.add_conditional_edges("intent", route_intent,
                            {"extract": "extract", "greeting": "greeting", "rag": "rag", "general": "general"})
    g.add_conditional_edges("extract", route_after_extract,
                            {"decline": "decline", "tool": "tool", "rag": "rag", "lead": "lead", "confirm": "confirm"})
    for n in ("greeting", "general", "rag", "lead", "confirm", "tool", "decline"):
        g.add_edge(n, END)
    return g.compile()


def run_turn(app, state: AgentState, user_input: str) -> tuple[AgentState, dict]:
    """Run one user turn; returns the new state and per-turn telemetry."""
    llm.reset_log()
    t0 = time.perf_counter()
    out = app.invoke({**state, "user_input": user_input, "trace": [], "declined": False, "new_fields": [],
                      "affirmed": False, "negated": False, "response": "", "retrieved": [], "query": None})
    latency = time.perf_counter() - t0
    out["history"] = state.get("history", []) + [{"role": "user", "content": user_input},
                                                 {"role": "assistant", "content": out["response"]}]
    return out, {"latency_s": latency, "trace": out.get("trace", []), **llm.summarize(llm.get_log())}


class AgentSystem:
    """Uniform conversational interface used by the evaluation scripts."""

    def __init__(self, retriever: Retriever, sticky: bool = True, interrupt: bool = True, confirm: bool = True):
        self.app = build_graph(retriever, sticky=sticky, interrupt=interrupt, confirm=confirm)
        self.reset()

    def reset(self):
        self.state = new_state()

    def turn(self, text: str) -> tuple[str, dict]:
        self.state, tel = run_turn(self.app, self.state, text)
        tel["retrieved"] = self.state.get("retrieved", [])
        tel["intent"] = self.state.get("intent")
        return self.state["response"], tel

    @property
    def registrations(self) -> list[dict]:
        return self.state.get("crm_records", [])
