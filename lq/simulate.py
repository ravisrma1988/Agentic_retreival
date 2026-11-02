"""LLM user simulator: plays a persona against a system and records the full dialogue."""
import re

from . import config, llm, prompts

_PREFIX_RE = re.compile(r"^\s*(?:user|you|me|customer)\s*:\s*", re.I)


def _clean(text: str) -> str:
    text = _PREFIX_RE.sub("", text.strip().splitlines()[0] if text.strip() else "")
    return text.strip().strip('"').strip()


def simulate(system, persona: dict, max_turns: int = config.SIM_MAX_TURNS) -> dict:
    system.reset()
    behaviour = prompts.SIM_BEHAVIOURS[persona["behaviour"]].format(**persona)
    sim_sys = prompts.SIM_SYSTEM.format(**{**persona, "behaviour": behaviour})
    # Role-flipped transcript from the simulator's point of view.
    sim_msgs = [{"role": "system", "content": sim_sys},
                {"role": "user", "content": "(The chat window has just opened. Write your first message.)"}]
    turns, end_reason = [], "max_turns"
    for t in range(max_turns):
        llm.reset_log()
        raw = llm.chat(sim_msgs, node="simulator", aux=True, temperature=config.SIM_TEMPERATURE,
                       max_tokens=80, seed=persona["seed"] * 100 + t)
        if "[END]" in raw.upper() or not _clean(raw):
            end_reason = "sim_end"
            break
        user = _clean(raw)
        reply, tel = system.turn(user)
        turns.append({"t": t, "user": user, "bot": reply, "n_registrations": len(system.registrations), **tel})
        sim_msgs += [{"role": "assistant", "content": user}, {"role": "user", "content": reply}]
    return {"persona": persona, "turns": turns, "registrations": list(system.registrations), "end_reason": end_reason}
