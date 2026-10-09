from __future__ import annotations
 
import json
from dataclasses import dataclass, field
from typing import Any

from mlflow.entities import SpanType

from app import observability
from app.agent.tools import ToolRegistry
from app.llm.client import BaseLLMClient, get_llm_client
 
MAX_ITERATIONS = 4

# How many past exchanges stay in context for follow-ups like "what about
# Q2?". Each turn is kept whole -- question, tool calls, tool results, final
# answer -- because small local models (Qwen 7B) otherwise learn from the
# bare Q&A history that answers appear without tool calls, and start
# inventing figures on the second turn. Tool results here are a few numbers,
# so six full turns stay well within context.
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
- When comparing periods, pass the earlier period as period_a and the later \
as period_b; "change" and "direction" describe the move from period_a to \
period_b.
- Earlier answers in this conversation are not a data source. Every question \
that needs a figure, including follow-ups like "what about Q2?" or "how does \
that compare?", requires a fresh tool call in this turn before you answer.
 
YOUR ANSWER WILL BE READ ALOUD. So:
- Plain sentences only. No markdown, no bullet points, no headings, no \
asterisks, no tables.
- Keep it to one to three sentences. A listener cannot re-read.
- Round naturally when speaking, and get the unit right: 19940 is "about \
20 thousand dollars", 405200 is "about 405 thousand dollars", 1779000 is \
"about 1.8 million dollars". State exact figures only when the user asks \
for exact numbers.
- Never read out a date as "2025-03". Say "March".
 
{context}
- For questions about recent events, current regulations, or anything outside the local dataset, call search_web immediately without asking permission. Never say "would you like me to search" — just search."""
 
 
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
        client: BaseLLMClient | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.client = client or get_llm_client()
        self.registry = registry or ToolRegistry()
        # Messages from past turns (user, assistant tool calls, tool results,
        # final assistant answer), so "what about Q2?" resolves against what
        # was just discussed AND the model keeps seeing that figures come
        # from tool calls. Trimmed to MAX_HISTORY_TURNS whole turns.
        self.history: list[dict[str, Any]] = []

    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=self.registry.system_context()),
        }

    def reset_history(self) -> None:
        """Start a fresh conversation -- e.g. between independent eval cases."""
        self.history = []

    def _remember(self, turn_messages: list[dict[str, Any]], answer: str) -> None:
        """Keep this turn's messages (user question through tool results)
        plus the final answer, then drop the oldest whole turns beyond
        MAX_HISTORY_TURNS -- a turn starts at each user message."""
        self.history.extend(turn_messages)
        self.history.append({"role": "assistant", "content": answer})
        starts = [i for i, m in enumerate(self.history) if m.get("role") == "user"]
        if len(starts) > MAX_HISTORY_TURNS:
            self.history = self.history[starts[-MAX_HISTORY_TURNS]:]

    @observability.trace(name="agent.ask", span_type=SpanType.AGENT)
    def ask(self, question: str) -> AgentTurn:
        turn = AgentTurn(question=question)
        messages: list[dict[str, Any]] = [
            self._system_message(),
            *self.history,
            {"role": "user", "content": question},
        ]
        turn_start = len(messages) - 1

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
                self._remember(messages[turn_start:], turn.answer)
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
        self._remember(messages[turn_start:], turn.answer)
        return turn

    def close(self) -> None:
        self.client.close()
 
    def __enter__(self) -> "FinancialAgent":
        return self
 
    def __exit__(self, *exc: object) -> None:
        self.close()
 
