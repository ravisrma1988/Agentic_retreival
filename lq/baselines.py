"""Baselines sharing the AgentSystem interface (reset / turn / registrations).

  llm_only     -- ungrounded LLM (the original paper's baseline)
  full_context -- whole knowledge base in the system prompt ("prompt stuffing")
  naive_rag    -- single-step retrieve-then-generate with the same retriever, no routing or query rewriting
  react        -- LangGraph prebuilt ReAct agent with search and register_lead tools (instruction-gated tool use)
  react_confirm -- the same ReAct agent with a LangGraph human-in-the-loop interrupt: register_lead runs only after
                   the user confirms the details read back to them
"""
import time
import uuid

from . import config, llm, prompts
from .kb import full_kb_text
from .retriever import Retriever, format_context


class _ChatBaseline:
    def __init__(self):
        self.reset()

    def reset(self):
        self.history: list[dict] = []

    @property
    def registrations(self) -> list:
        return []  # these baselines have no lead tool

    def _system(self, text: str) -> tuple[str, list]:
        raise NotImplementedError

    def turn(self, text: str) -> tuple[str, dict]:
        llm.reset_log()
        t0 = time.perf_counter()
        system, retrieved = self._system(text)
        reply = llm.chat([{"role": "system", "content": system}, *self.history[-6:],
                          {"role": "user", "content": text}], node=type(self).__name__)
        latency = time.perf_counter() - t0
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        return reply, {"latency_s": latency, "retrieved": retrieved, **llm.summarize(llm.get_log())}


class LLMOnly(_ChatBaseline):
    def _system(self, text):
        return prompts.LLM_ONLY_SYSTEM, []


class FullContext(_ChatBaseline):
    def __init__(self):
        self.kb = full_kb_text()
        super().__init__()

    def _system(self, text):
        return prompts.FULL_CONTEXT_SYSTEM.format(context=self.kb), []


class NaiveRAG(_ChatBaseline):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        super().__init__()

    def _system(self, text):
        hits = self.retriever.search(text)
        return prompts.RAG_SYSTEM.format(context=format_context(hits)), [h["chunk_id"] for h in hits]


class ReactBaseline:
    def __init__(self, retriever: Retriever):
        from langchain_core.tools import tool
        from langchain_openai import ChatOpenAI

        self.retriever = retriever
        self._last_hits: list[str] = []

        @tool
        def search_knowledge_base(query: str) -> str:
            """Search AutoStream's knowledge base (plans, pricing, features, policies, apps, support)."""
            hits = retriever.search(query)
            self._last_hits += [h["chunk_id"] for h in hits]
            return format_context(hits)

        @tool
        def register_lead(name: str, email: str, platform: str) -> str:
            """Register a sales lead in the CRM. Call only after the user has given their name, email and content platform."""
            self.registrations.append({"name": name, "email": email, "platform": platform, "turn": self._turn_idx})
            return "Lead registered successfully."

        self.model = ChatOpenAI(model=config.AGENT_MODEL, base_url=config.AGENT_BASE_URL, api_key="EMPTY",
                                temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS, seed=config.SEED)
        self.tools = [search_knowledge_base, register_lead]
        self.agent = self._build()
        self.reset()

    def _build(self, **kw):
        from langgraph.prebuilt import create_react_agent
        try:
            return create_react_agent(self.model, self.tools, prompt=prompts.REACT_SYSTEM, **kw)
        except TypeError:  # older langgraph releases
            return create_react_agent(self.model, self.tools, state_modifier=prompts.REACT_SYSTEM, **kw)

    def reset(self):
        self.messages: list = []
        self.registrations: list[dict] = []
        self._turn_idx = 0

    def turn(self, text: str) -> tuple[str, dict]:
        from langchain_core.messages import AIMessage, HumanMessage

        llm.reset_log()
        self._last_hits = []
        t0 = time.perf_counter()
        error = None
        try:
            out = self.agent.invoke({"messages": self.messages + [HumanMessage(text)]},
                                    config={"recursion_limit": 12})
            msgs = out["messages"]
        except Exception as e:  # recursion limit or malformed tool call: count as a failed turn
            error = f"{type(e).__name__}: {e}"
            msgs = self.messages + [HumanMessage(text), AIMessage("")]
        latency = time.perf_counter() - t0
        new = msgs[len(self.messages) + 1:]
        for m in new:
            if isinstance(m, AIMessage):
                u = getattr(m, "usage_metadata", None) or {}
                llm.log_event("react", 0.0, u.get("input_tokens", 0), u.get("output_tokens", 0))
        self.messages = msgs
        self._turn_idx += 1
        reply = next((m.content for m in reversed(new) if isinstance(m, AIMessage) and m.content), "")
        tel = {"latency_s": latency, "retrieved": self._last_hits, "error": error, **llm.summarize(llm.get_log())}
        tel["llm_latency_s"] = max(latency - tel["retrieval_latency_s"], 0.0)
        return reply, tel


class ReactConfirmBaseline(ReactBaseline):
    """ReAct with LangGraph's human-in-the-loop interrupt before tool execution. Search calls are resumed at once;
    a register_lead call pauses the run, its arguments are read back to the user, and the call executes only if the
    user's next message confirms them. Any other reply is returned to the agent as the tool result."""

    def _build(self, **kw):
        from langgraph.checkpoint.memory import MemorySaver
        return super()._build(checkpointer=MemorySaver(), interrupt_before=["tools"], **kw)

    def reset(self):
        super().reset()
        self.thread = uuid.uuid4().hex
        self.pending: list = []
        self._n_seen = 0

    def turn(self, text: str) -> tuple[str, dict]:
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
        from .extractor import extract_fields, is_affirmative

        llm.reset_log()
        self._last_hits = []
        t0 = time.perf_counter()
        cfg = {"configurable": {"thread_id": self.thread}, "recursion_limit": 12}
        error, reply = None, ""
        try:
            if self.pending:
                calls, self.pending = self.pending, []
                if is_affirmative(text) and not any(extract_fields(text, None).values()):
                    self.agent.invoke(None, cfg)  # approved: execute the paused tool calls
                else:
                    rejected = [ToolMessage(prompts.REACT_CONFIRM_REJECT.format(reply=text), tool_call_id=c["id"])
                                for c in calls]
                    self.agent.update_state(cfg, {"messages": rejected}, as_node="tools")
                    self.agent.invoke(None, cfg)
            else:
                self.agent.invoke({"messages": [HumanMessage(text)]}, cfg)
            for _ in range(6):  # resume search calls; stop at a lead write
                state = self.agent.get_state(cfg)
                if not state.next:
                    break
                calls = getattr(state.values["messages"][-1], "tool_calls", None) or []
                lead = [c for c in calls if c["name"] == "register_lead"]
                if lead:
                    a = lead[-1]["args"]
                    self.pending = calls
                    reply = prompts.CONFIRM_ASK.format(name=a.get("name", ""), email=a.get("email", ""),
                                                       platform=a.get("platform", ""))
                    break
                self.agent.invoke(None, cfg)
        except Exception as e:  # recursion limit or malformed tool call: count as a failed turn, start a fresh thread
            error = f"{type(e).__name__}: {e}"
            self.pending, self.thread, self._n_seen = [], uuid.uuid4().hex, 0
        latency = time.perf_counter() - t0
        msgs = self.agent.get_state({"configurable": {"thread_id": self.thread}}).values.get("messages", [])
        new = msgs[self._n_seen:]
        self._n_seen = len(msgs)
        for m in new:
            if isinstance(m, AIMessage):
                u = getattr(m, "usage_metadata", None) or {}
                llm.log_event("react", 0.0, u.get("input_tokens", 0), u.get("output_tokens", 0))
        if not reply:
            reply = next((m.content for m in reversed(new) if isinstance(m, AIMessage) and m.content), "")
        self._turn_idx += 1
        tel = {"latency_s": latency, "retrieved": self._last_hits, "error": error, **llm.summarize(llm.get_log())}
        tel["llm_latency_s"] = max(latency - tel["retrieval_latency_s"], 0.0)
        return reply, tel


def make_system(name: str, retriever: Retriever | None):
    from .agent import AgentSystem

    factories = {
        "agentic_rag": lambda: AgentSystem(retriever),
        "agentic_rag_no_sticky": lambda: AgentSystem(retriever, sticky=False),
        "agentic_rag_no_interrupt": lambda: AgentSystem(retriever, interrupt=False),
        "agentic_rag_no_confirm": lambda: AgentSystem(retriever, confirm=False),
        "llm_only": LLMOnly,
        "full_context": FullContext,
        "naive_rag": lambda: NaiveRAG(retriever),
        "react": lambda: ReactBaseline(retriever),
        "react_confirm": lambda: ReactConfirmBaseline(retriever),
    }
    return factories[name]()
