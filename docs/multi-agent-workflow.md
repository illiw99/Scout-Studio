# Efficient automatic development agents

Start a Codex CLI or IDE session in this repository and describe the task.
`AGENTS.md` selects the least expensive useful workflow. The coordinator normally
investigates, implements, checks and self-reviews. Existing model, reasoning,
authentication and permissions are inherited. No background service is started;
project configuration may need to be trusted before the client loads it.

| Task | Default delegation |
| --- | --- |
| Questions, docs, small fixes and straightforward features | None |
| Difficult investigation | One focused researcher if useful |
| Substantial independent pipeline and native UI work | One worker; coordinator owns the other area |
| Media overwrite/access, review integrity, security or broad contracts | One independent reviewer |

Six optional roles live in `.codex/agents/`: `pipeline`, `desktop`,
`infrastructure`, `researcher`, `verifier`, and `reviewer`. They are tools, not six
mandatory stages. Default to at most one child; use two only for independent
substantial work. Configuration caps concurrent child threads at two;
AGENTS.md limits each user task to two child threads total, including replacements,
unless the user requests more. Children never spawn children. Reuse the existing
thread for at most one delegated correction round, then resolve locally or report
the blocker. These controls are not a hard token or cost limit.

Read the [code map](code-map.md), then inspect targeted source. Exclude footage,
weights, caches, virtual environments and generated output from broad scans.
Reuse established findings and successful checks while their inputs remain
unchanged. Avoid full-history handoffs and raw log dumps. The coordinator runs
known checks directly and delegates verification only when diagnosis is substantial.

Assignments include objective, acceptance criteria, owned files, allowed
operations (read/edit/check), dependency inputs and expected return. Returns give
status (done/blocked), files inspected/changed, actionable findings, actual command
results and remaining risks, normally within 200 words. Shared files have one
writer; agree on contracts and hand off ownership explicitly. Review/check a
stable result after writers finish. Read-only roles are instruction-constrained;
they inherit parent permissions without an extra OS sandbox or file lock.

Keep dependencies, ownership, status and correction count in the coordinator's
plan. Long tasks may use ignored `.codex/runs/<task>.md` ledgers without secrets,
footage or personal data. If delegation is unavailable, work sequentially and
report that limitation. The coordinator makes focused local commits after coherent
changes and relevant checks, preserves unrelated edits, and reports hashes.
Workers never stage or commit. Pushing/publishing still needs user authorization.

## Validation

Run `python3 scripts/check-agents.py` for offline TOML/schema checks. Python 3.11+
provides `tomllib`; Python 3.10 can use optional development package `tomli`
(`python3 -m pip install tomli`). This does not change application dependencies.
Run `bash scripts/check.sh` for agent validation, CLI help and unittest discovery;
use `PYTHON=.venv/bin/python` to select an installed environment. Desktop tests
need `pip install -e '.[desktop]'`; skipped Qt tests do not prove UI behavior.
Automated checks default to offscreen Qt. Real playback/audio, display behavior,
codecs and hardware acceleration require appropriate native smoke testing.

In supported CLI sessions, `/subagents` shows delegated work. Explicit smoke test:
“Use one researcher to trace review save conflict detection without editing.”
Normal small tasks should create no child threads. Never smoke-test against real
footage, field files, trained models, caches or user-edited MLT reviews.

The source workflow's optional Graphify indexing is not installed here. Start with
the short code map and scoped searches. Reconsider indexing only when repeated
tracing warrants its indexing, refresh and query cost; exclude user media, secrets,
models, caches and generated files, and verify findings against current source.

Configuration: [official OpenAI subagent docs](https://learn.chatgpt.com/docs/agent-configuration/subagents).
Patterns: [JetBrains agentic workflows](https://www.jetbrains.com/pages/ai-agents/architecture/agentic-workflows/).
