from __future__ import annotations
 
import json
from dataclasses import dataclass, field
from typing import Any
 
from app.agent.tools import ToolRegistry
from app.llm.client import LLMClient
 
MAX_ITERATIONS = 4
 
SYSTEM_PROMPT = """You are a financial assistant answering questions about \
company revenue by voice.
 
CRITICAL RULES:
- Never state a figure that did not come from a tool result. You have no \
revenue knowledge of your own.
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
 
    def _system_message(self) -> dict[str, str]:
        return {
            "role": "system",
            "content": SYSTEM_PROMPT.format(context=self.registry.system_context()),
        }
 
    def ask(self, question: str) -> AgentTurn:
        turn = AgentTurn(question=question)
        messages: list[dict[str, Any]] = [
            self._system_message(),
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
        return turn
 
    def close(self) -> None:
        self.client.close()
 
    def __enter__(self) -> "FinancialAgent":
        return self
 
    def __exit__(self, *exc: object) -> None:
        self.close()
 
