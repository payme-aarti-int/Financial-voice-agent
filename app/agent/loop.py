from __future__ import annotations
 
import json
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import SpanType

from app import observability
from app.agent.tools import ToolRegistry
from app.llm.client import LLMClient
 
MAX_ITERATIONS = 4

# How many past exchanges (user question + final answer) stay in context for
# follow-ups like "what about Q2?". Intermediate tool-call messages within a
# single ask() are NOT kept across turns -- only the final Q&A -- so history
# doesn't balloon with stale tool JSON.
MAX_HISTORY_TURNS = 6
 
SYSTEM_PROMPT = """You are a financial assistant answering questions by voice.
You have access to two data sources:
1. Monthly company P&L data (revenue, EBITDA, margins, costs) from Jan 2024 to Dec 2025.
2. RBI weekly reserve money data (currency in circulation, bankers deposits, net forex assets) from July 2001 to August 2020.
 
CRITICAL RULES:
- Never state a figure that did not come from a tool result. You have no financial knowledge of your own.
- If a tool returns "available": false, say plainly that you do not have data \
for that period, and state what period you do have. Never estimate, never \
extrapolate, never substitute a nearby month.
- Do not perform arithmetic yourself. If you need a total or a change, call \
the tool that computes it.
 
YOUR ANSWER WILL BE READ ALOUD. So:
- Plain sentences only. No markdown, no bullet points, no headings, no \
asterisks, no tables.
- Keep it to one to three sentences. A listener cannot re-read.
- Round naturally when speaking: 1779000 becomes "about 1.8 million dollars". \
State exact figures only when the user asks for exact numbers.
- Never read out a date as "2025-03". Say "March".
 
{context}"""
 
 
@dataclass
class AgentTurn:
 
    question: str
    answer: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    hit_iteration_limit: bool = False
 
 
class FinancialAgent:
    def __init__(
        self,
        client: LLMClient | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.client = client or LLMClient()
        self.registry = registry or ToolRegistry()
        # Alternating user/assistant messages from past turns, so "what
        # about Q2?" resolves against what was just discussed. Only the
        # final Q&A per turn is kept -- not the tool-call messages inside
        # a single ask() -- so this doesn't balloon with stale tool JSON.
        self.history: list[dict[str, str]] = []

    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=self.registry.system_context()),
        }

    def reset_history(self) -> None:
        """Start a fresh conversation -- e.g. between independent eval cases."""
        self.history = []

    def _remember(self, question: str, answer: str) -> None:
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    @observability.trace(name="agent.ask", span_type=SpanType.AGENT)
    def ask(self, question: str) -> AgentTurn:
        turn = AgentTurn(question=question)
        messages: list[dict[str, Any]] = [
            self._system_message(),
            *self.history,
            {"role": "user", "content": question},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            turn.iterations = iteration

            completion = self.client.complete(
                messages,
                tools=self.registry.schemas,
            )
            message = completion.raw["choices"][0]["message"]
            requested = message.get("tool_calls") or []

            if not requested:
                turn.answer = (message.get("content") or "").strip()
                self._remember(question, turn.answer)
                return turn

            messages.append(message)

            for call in requested:
                name = call["function"]["name"]
                raw_arguments = call["function"].get("arguments", "")
                result = self.registry.execute(name, raw_arguments)

                turn.tool_calls.append(
                    {"name": name, "arguments": raw_arguments, "result": result}
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "name": name,
                        "content": json.dumps(result),
                    }
                )

        turn.hit_iteration_limit = True
        turn.answer = (
            "I could not work that out reliably. Could you rephrase the question?"
        )
        self._remember(question, turn.answer)
        return turn

    def close(self) -> None:
        self.client.close()
 
    def __enter__(self) -> "FinancialAgent":
        return self
 
    def __exit__(self, *exc: object) -> None:
        self.close()
 
