"""Baselines sharing the AgentSystem interface (reset / turn / registrations).

  llm_only     -- ungrounded LLM (the original paper's baseline)
  full_context -- whole knowledge base in the system prompt ("prompt stuffing")
  naive_rag    -- single-step retrieve-then-generate with the same retriever, no routing or query rewriting
  react        -- LangGraph prebuilt ReAct agent with search and register_lead tools (instruction-gated tool use)
"""
import time

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
        from langgraph.prebuilt import create_react_agent

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

        model = ChatOpenAI(model=config.AGENT_MODEL, base_url=config.AGENT_BASE_URL, api_key="EMPTY",
                           temperature=config.TEMPERATURE, max_tokens=config.MAX_TOKENS, seed=config.SEED)
        tools = [search_knowledge_base, register_lead]
        try:
            self.agent = create_react_agent(model, tools, prompt=prompts.REACT_SYSTEM)
        except TypeError:  # older langgraph releases
            self.agent = create_react_agent(model, tools, state_modifier=prompts.REACT_SYSTEM)
        self.reset()

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
    }
    return factories[name]()
