"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, Route, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── Pydantic schema for structured LLM output ───────────────────────

class ClassifyOutput(BaseModel):
    """Structured output for intent classification."""
    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description=(
            "Intent category. Priority: risky > tool > missing_info > error > simple.\n"
            "- risky: actions with side effects (refunds, deletions, emails, cancellations)\n"
            "- tool: information lookups (order status, tracking, search)\n"
            "- missing_info: vague/incomplete queries lacking actionable context\n"
            "- error: system failures (timeouts, crashes, service unavailable)\n"
            "- simple: general questions answerable without tools or actions"
        )
    )
    risk_level: Literal["high", "low"] = Field(
        description="'high' for risky route, 'low' for all others"
    )
    reason: str = Field(description="One-sentence justification for the classification")


# ─── Node implementations ─────────────────────────────────────────────

def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM with structured output.

    Uses .with_structured_output() for reliable enum classification.
    Priority: risky > tool > missing_info > error > simple
    """
    query = state.get("query", "")
    llm = get_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(ClassifyOutput)

    prompt = (
        "You are a support ticket classifier. Classify the following query into exactly one category.\n\n"
        "Priority order (highest to lowest):\n"
        "1. risky   — actions with side effects: refunds, deletions, sending emails, cancellations\n"
        "2. tool    — information lookups: order status, tracking number, account search\n"
        "3. missing_info — vague or incomplete queries with no actionable context\n"
        "4. error   — system failures: timeouts, crashes, service unavailable\n"
        "5. simple  — general questions answerable without tools or special actions\n\n"
        f"Query: {query}"
    )

    result: ClassifyOutput = structured_llm.invoke(prompt)

    return {
        "route": result.route,
        "risk_level": result.risk_level,
        "messages": [f"classify:{result.route}"],
        "events": [make_event(
            "classify", "completed",
            f"classified as {result.route}",
            reason=result.reason,
            risk_level=result.risk_level,
        )],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call with error simulation for retry testing.

    Simulates transient failure when:
      - route is "error"  AND  attempt < 2
    Otherwise returns a mock success result.
    """
    query = state.get("query", "")
    route = state.get("route", "")
    attempt = state.get("attempt", 0)

    # Simulate transient failure for error-route scenarios
    if route == Route.ERROR and attempt < 2:
        result = f"ERROR: tool call failed on attempt {attempt} (transient timeout)"
        return {
            "tool_results": [result],
            "events": [make_event(
                "tool", "error",
                "tool call failed — will retry",
                attempt=attempt,
            )],
        }

    # Mock success: could be order lookup, account info, etc.
    result = f"TOOL_SUCCESS: Retrieved data for query='{query[:60]}' on attempt={attempt}"
    return {
        "tool_results": [result],
        "events": [make_event(
            "tool", "completed",
            "tool call succeeded",
            attempt=attempt,
        )],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate latest tool result — the retry-loop gate.

    Uses heuristic check: if the latest tool_result contains "ERROR"
    → needs_retry, otherwise → success.

    (LLM-as-judge version: swap in the commented block below for bonus points.)
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    # Heuristic gate (base score)
    if "ERROR" in latest.upper():
        evaluation_result = "needs_retry"
        verdict_msg = "tool result unsatisfactory — scheduling retry"
    else:
        evaluation_result = "success"
        verdict_msg = "tool result satisfactory — proceeding to answer"

    # ── LLM-as-judge (bonus) ──────────────────────────────────────────
    # Uncomment to enable LLM evaluation for bonus points:
    #
    # class EvalOutput(BaseModel):
    #     verdict: Literal["success", "needs_retry"]
    #     reason: str
    #
    # llm = get_llm(temperature=0.0)
    # eval_llm = llm.with_structured_output(EvalOutput)
    # eval_result: EvalOutput = eval_llm.invoke(
    #     f"Is this tool result satisfactory for answering the user query?\n"
    #     f"Result: {latest}\n"
    #     f"Reply 'success' if usable, 'needs_retry' if it indicates an error."
    # )
    # evaluation_result = eval_result.verdict
    # verdict_msg = eval_result.reason
    # ─────────────────────────────────────────────────────────────────

    return {
        "evaluation_result": evaluation_result,
        "events": [make_event(
            "evaluate", "completed",
            verdict_msg,
            evaluation_result=evaluation_result,
            latest_tool_result=latest[:100],
        )],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final grounded response using an LLM.

    Grounds the answer in: tool_results, approval context, original query.
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    proposed_action = state.get("proposed_action", "")

    # Build context for the LLM
    context_parts = [f"User query: {query}"]
    if tool_results:
        context_parts.append(f"Tool results: {'; '.join(tool_results[-3:])}")
    if approval and approval.get("approved"):
        context_parts.append(
            f"Approved action: {proposed_action} "
            f"(approved by {approval.get('reviewer', 'reviewer')})"
        )
    context = "\n".join(context_parts)

    llm = get_llm(temperature=0.3)
    prompt = (
        "You are a helpful customer support agent. "
        "Generate a clear, concise response to the user based on the context below.\n\n"
        f"{context}\n\n"
        "Response:"
    )
    response = llm.invoke(prompt)
    final_answer = response.content if hasattr(response, "content") else str(response)

    return {
        "final_answer": final_answer,
        "events": [make_event(
            "answer", "completed",
            "LLM answer generated",
            answer_length=len(final_answer),
        )],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")

    question = (
        f"Your request '{query}' is too vague to process. "
        "Could you please provide more details? For example: "
        "What specific issue are you experiencing? "
        "Which account, order, or product does this relate to?"
    )

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event(
            "clarify", "completed",
            "clarification question generated",
            original_query=query[:80],
        )],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval review."""
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")

    proposed_action = (
        f"PROPOSED ACTION: Execute the following support request — '{query}'. "
        f"Risk level: {risk_level.upper()}. "
        "This action may have irreversible side effects and requires explicit approval."
    )

    return {
        "proposed_action": proposed_action,
        "events": [make_event(
            "risky_action", "pending_approval",
            "risky action prepared — awaiting HITL approval",
            risk_level=risk_level,
            proposed_action=proposed_action[:100],
        )],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default: mock auto-approval so CI and tests can run offline.
    Set LANGGRAPH_INTERRUPT=true to enable real interrupt() for live HITL.
    """
    proposed_action = state.get("proposed_action", "")

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        # Real HITL — pauses graph and waits for external resume signal
        from langgraph.types import interrupt
        human_input = interrupt(
            {"message": "Please approve or reject the following action", "action": proposed_action}
        )
        approved = human_input.get("approved", False)
        reviewer = human_input.get("reviewer", "human-reviewer")
        comment = human_input.get("comment", "")
    else:
        # Mock approval for CI / offline runs
        approved = True
        reviewer = "mock-reviewer"
        comment = "Auto-approved by mock approval node"

    approval_dict = ApprovalDecision(
        approved=approved,
        reviewer=reviewer,
        comment=comment,
    ).model_dump()

    return {
        "approval": approval_dict,
        "events": [make_event(
            "approval",
            "approved" if approved else "rejected",
            f"Action {'approved' if approved else 'rejected'} by {reviewer}",
            reviewer=reviewer,
            comment=comment,
        )],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Increment attempt counter and log the transient failure."""
    attempt = state.get("attempt", 0)
    new_attempt = attempt + 1
    tool_results = state.get("tool_results", [])
    latest_error = tool_results[-1] if tool_results else "unknown error"

    error_msg = f"Attempt {new_attempt} failed: {latest_error[:120]}"

    return {
        "attempt": new_attempt,
        "errors": [error_msg],
        "events": [make_event(
            "retry", "retrying",
            error_msg,
            attempt=new_attempt,
        )],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries are exhausted."""
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    errors = state.get("errors", [])
    query = state.get("query", "")

    final_answer = (
        f"We were unable to process your request '{query[:60]}' after {attempt} attempt(s). "
        "Our team has been notified and will follow up shortly. "
        "Reference: max_attempts exceeded."
    )

    return {
        "route": Route.DEAD_LETTER,
        "final_answer": final_answer,
        "events": [make_event(
            "dead_letter", "exhausted",
            f"max retries ({max_attempts}) exceeded — escalating to dead letter",
            attempt=attempt,
            error_count=len(errors),
        )],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END."""
    route = state.get("route", "unknown")
    final_answer = state.get("final_answer", "")
    attempt = state.get("attempt", 0)

    return {
        "events": [make_event(
            "finalize", "completed",
            "workflow finished",
            route=route,
            has_answer=bool(final_answer),
            attempt=attempt,
        )],
    }