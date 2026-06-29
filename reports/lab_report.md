# Day 08 Lab Report

## 1. Team / student

- Name: Lê Quang Miền — 2A202600715
- Repo/commit: https://github.com/LeQuangMien/phase2-track3-day8-langgraph-agent
- Date: 2026-06-29

## 2. Architecture

The graph implements a support-ticket routing agent with 11 nodes wired into a `StateGraph(AgentState)`. All routing is driven by real LLM classification — no hardcoded keyword matching.

**Graph flow:**
```
START → intake → classify → [route_after_classify]
  simple       → answer → finalize → END
  tool         → tool → evaluate → [route_after_evaluate]
                                     success     → answer → finalize → END
                                     needs_retry → retry  → [route_after_retry]
                                                              attempt < max → tool (loop)
                                                              exhausted    → dead_letter → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → [route_after_approval]
                                             approved → tool → evaluate → ...
                                             rejected → clarify → finalize → END
  error        → retry → [route_after_retry] → ...
```

**Nodes (11 total):**

| Node | Role |
|---|---|
| `intake` | Normalize raw query string |
| `classify` | LLM + `.with_structured_output(ClassifyOutput)` → route + risk_level |
| `tool` | Mock tool call; simulates transient ERROR when route=error and attempt<2 |
| `evaluate` | Heuristic gate: "ERROR" in result → needs_retry, else success |
| `answer` | LLM-grounded final response using tool_results and approval context |
| `clarify` | Generate clarification question for vague/incomplete queries |
| `risky_action` | Prepare proposed action string for HITL review |
| `approval` | Mock approval (auto-approved); LANGGRAPH_INTERRUPT=true enables real interrupt() |
| `retry` | Increment attempt counter, append error log |
| `dead_letter` | Handle max-retry exhaustion, write final_answer explaining failure |
| `finalize` | Emit final audit event; all routes converge here before END |

**Conditional edges (4 routing functions):**

| From | Routing function | Possible destinations |
|---|---|---|
| `classify` | `route_after_classify` | answer / tool / clarify / risky_action / retry |
| `evaluate` | `route_after_evaluate` | answer / retry |
| `retry` | `route_after_retry` | tool / dead_letter |
| `approval` | `route_after_approval` | tool / clarify |

## 3. State schema

List important fields and whether they are overwrite or append-only.

| Field | Reducer | Why |
|---|---|---|
| `messages` | append | audit conversation/events |
| `tool_results` | append | all tool call results; evaluate reads latest |
| `errors` | append | all retry error messages accumulated |
| `events` | append | full node-level audit log for grading |
| `route` | overwrite | current route only |
| `risk_level` | overwrite | high / low, set once by classify |
| `attempt` | overwrite | incremented by retry node |
| `max_attempts` | overwrite | per-scenario limit, set at init |
| `final_answer` | overwrite | last answer wins |
| `evaluation_result` | overwrite | latest evaluate verdict only |
| `pending_question` | overwrite | latest clarification question |
| `proposed_action` | overwrite | latest risky action description |
| `approval` | overwrite | latest HITL decision dict |

## 4. Scenario results

Paste the key metrics from `outputs/metrics.json`.

**Summary:** 7 scenarios · success rate 85.71% · avg nodes visited 6.4 · total retries 3 · total interrupts 2

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | Yes | 0 | 0 |
| S02_tool | tool | tool | Yes | 0 | 0 |
| S03_missing | missing_info | missing_info | Yes | 0 | 0 |
| S04_risky | risky | risky | Yes | 0 | 1 |
| S05_error | error | error | Yes | 2 | 0 |
| S06_delete | risky | risky | Yes | 0 | 1 |
| S07_dead_letter | error | dead_letter | No | 1 | 0 |

S07 (`max_attempts=1`) routes to `dead_letter` instead of `error` because the retry loop exhausts after a single attempt — this is correct bounded-retry behavior. The grader's strict route-string equality marks it as failed, but the graph logic is working as intended.

## 5. Failure analysis

Describe at least two failure modes you considered:

1. Retry or tool failure: If `route_after_retry` does not check `attempt >= max_attempts`, the graph loops forever between `tool → evaluate → retry → tool`. Mitigation: `route_after_retry` strictly compares `attempt < max_attempts` and routes to `dead_letter` when the limit is reached. S07 (`max_attempts=1`) stress-tests this path: it exhausts after one attempt and terminates cleanly via `dead_letter → finalize → END`.

2. Risky action without approval: If the `risky_action → approval → tool` path could bypass `approval_node`, irreversible side-effect actions (refunds, deletions, emails) would execute without human oversight. Mitigation: `risky_action_node` always sets `proposed_action` and the graph forces every risky route through `approval_node`. `route_after_approval` gates on `approval["approved"]`; a rejected decision routes to `clarify` instead of `tool`, so no destructive action runs without an explicit approval signal. For production deployments, setting `LANGGRAPH_INTERRUPT=true` replaces mock auto-approval with a real `interrupt()` call that pauses the graph until a human resumes it.

## 6. Persistence / recovery evidence

Checkpointer: SQLite via `langgraph-checkpoint-sqlite` with WAL journal mode, configured in `configs/lab.yaml` (`checkpointer: sqlite`).

Each scenario run is assigned a unique `thread_id` (e.g. `thread-S01_simple`). The SQLite file `outputs/checkpoints.sqlite` persists across process restarts. Re-invoking the graph with the same `thread_id` and checkpointer instance resumes from the last saved checkpoint, demonstrating crash-resume capability without re-running completed nodes.

State history evidence (from `graph.get_state_history(config)` for thread `thread-S01_simple`):
```
step=-1  initial state injected
step=0   intake node completed
step=1   classify node completed
step=2   answer node completed
step=3   finalize node completed
step=4   END reached
```
6 checkpoints saved per thread. File size after 7 scenario runs: ~319 KB.

## 7. Extension work

Describe any extension you completed: SQLite/Postgres, time travel, fan-out/fan-in, graph diagram, tracing.

- SQLite persistence: `SqliteSaver` with WAL mode implemented in `persistence.py`. Crash-resume verified via `graph.get_state_history()` returning 6 checkpoints per scenario thread.
- Mock HITL with real interrupt() fallback: `approval_node` auto-approves in CI mode and switches to LangGraph `interrupt()` when `LANGGRAPH_INTERRUPT=true` is set, pausing the graph for a real human decision.
- LLM-as-judge stub: `evaluate_node` contains a commented block using `.with_structured_output(EvalOutput)` that can be activated for bonus LLM evaluation instead of the heuristic ERROR-string check.

## 8. Improvement plan

If you had one more day, what would you productionize first?

1. Real HITL with Streamlit UI: wire `LANGGRAPH_INTERRUPT=true` with a minimal Streamlit panel that displays `proposed_action` and approve/reject buttons, then resumes the graph via `graph.invoke(None, config)` with the human decision injected into state.
2. Parallel fan-out with `Send()` API: for the `tool` route, fan out concurrent sub-tool calls (e.g. order lookup and customer profile fetch simultaneously) using LangGraph's `Send()` API, then merge results before `evaluate` to reduce latency on multi-source queries.
3. Latency instrumentation: record `time.perf_counter()` at the start and end of each node and write `latency_ms` into `LabEvent.metadata`, enabling per-node P50/P95 latency tracking in the metrics report instead of the current placeholder `0`.
4. LLM-as-judge evaluation: activate the commented block in `evaluate_node` and benchmark it against the heuristic baseline across all 7 scenarios to measure precision and recall on the `needs_retry` decision.
