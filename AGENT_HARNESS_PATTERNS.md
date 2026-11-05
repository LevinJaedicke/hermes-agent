# Agent Harness Patterns Worth Copying — lessons from Hermes Agent

This document pulls out the parts of Hermes Agent that make it a **good agent harness**: the
mechanisms around the model loop that make an agent remember, learn, stay cheap, stay safe,
and keep working on long tasks. Provider integrations, installers, messaging adapters and other
"reach" features are deliberately left out — this is about the patterns you can lift into your
own project.

Each section follows the same shape:

- **What it is** — the feature in one or two lines.
- **How Hermes does it** — the concrete mechanics, numbers and rules.
- **Why it works** — the design insight.
- **Copy it** — the minimum you need to reproduce it.
- **Source** — where to read the real implementation in this repo.

---

## Table of contents

1. [Design principles that shape everything](#1-design-principles-that-shape-everything)
2. [Prompt assembly with a cache-stable prefix](#2-prompt-assembly-with-a-cache-stable-prefix)
3. [Bounded, curated persistent memory (MEMORY.md / USER.md)](#3-bounded-curated-persistent-memory)
4. [Session store + session search (cross-session recall)](#4-session-store--session-search)
5. [Skills: procedural memory with progressive disclosure](#5-skills-procedural-memory-with-progressive-disclosure)
6. [The background self-improvement review (the learning loop)](#6-the-background-self-improvement-review)
7. [The curator: keeping the skill library healthy](#7-the-curator-keeping-the-skill-library-healthy)
8. [Pluggable memory providers with a lifecycle contract](#8-pluggable-memory-providers)
9. [Context compression](#9-context-compression)
10. [Prompt caching strategy](#10-prompt-caching-strategy)
11. [Context files, progressive subdirectory hints, @-references](#11-context-files-subdirectory-hints-and-references)
12. [Tool registry, toolsets and availability checks](#12-tool-registry-toolsets-and-availability-checks)
13. [Tool Search: deferring rarely used tools](#13-tool-search-deferring-rarely-used-tools)
14. [Programmatic tool calling (`execute_code`)](#14-programmatic-tool-calling-execute_code)
15. [Subagent delegation](#15-subagent-delegation)
16. [File-editing safety: read tracking, stale-write guards, fuzzy patching](#16-file-editing-safety)
17. [Tool output limits and spill-to-file](#17-tool-output-limits-and-spill-to-file)
18. [Filesystem checkpoints (shadow git)](#18-filesystem-checkpoints-shadow-git)
19. [Dangerous-command approval](#19-dangerous-command-approval)
20. [Prompt-injection and secret hygiene](#20-prompt-injection-and-secret-hygiene)
21. [The turn loop: interrupts, budgets, retries, fallbacks, guards](#21-the-turn-loop-robustness)
22. [Background processes with completion notifications](#22-background-processes-with-completion-notifications)
23. [Planning and interaction tools: todo, clarify](#23-planning-and-interaction-tools)
24. [Long-running work: /goal, /loop, /heartbeat, cron, kanban](#24-long-running-work)
25. [Steering and conversation control: /steer, /queue, /btw, /undo, /rewind, /branch](#25-steering-and-conversation-control)
26. [Lifecycle hooks and plugin surfaces](#26-lifecycle-hooks-and-plugin-surfaces)
27. [Auxiliary models for side tasks](#27-auxiliary-models-for-side-tasks)
28. [Profiles: isolated agent identities](#28-profiles-isolated-agent-identities)
29. [Prompt-surface snapshot tests](#29-prompt-surface-snapshot-tests)
30. [Suggested adoption order](#30-suggested-adoption-order)

---

## 1. Design principles that shape everything

Two invariants run through the whole Hermes codebase. Adopt them first — most of the patterns
below are consequences of them.

**1. The conversation's cached prefix is sacred.**
Provider prompt caches (Anthropic `cache_control`, OpenAI automatic prefix caching, etc.) only
hit when the request starts with byte-identical content. So Hermes never:

- rebuilds the system prompt mid-conversation,
- swaps the tool list mid-conversation,
- reloads memory into the prompt mid-conversation,
- edits past messages.

The one allowed exception is **context compression**, which deliberately rebuilds. Anything
that would change system-prompt state (installing a skill, editing memory, changing the tool
set) takes effect at the **next session** — with an explicit opt-in `--now` for users who
accept the cache miss. Information that must arrive *during* a session is delivered through
the **per-turn user message** or **tool results**, never the system prompt.

**2. The core is a narrow waist; capability lives at the edges.**
Every tool schema is sent on every API call, so each core tool has a permanent token cost and
dilutes the model's attention. New capability should arrive as (in order of preference):
extending existing code → a CLI command + a skill → a tool gated by an availability check →
a plugin → an MCP server → *only last* a new core tool.

**Copy it:** write both rules into your contributor guide and review every PR against them.
They prevent the two most common harness failure modes: exploding cost from cache misses, and
a bloated tool list that makes the model worse.

---

## 2. Prompt assembly with a cache-stable prefix

**What it is:** a system prompt built once per session in ordered tiers, plus a separate
channel for per-turn ephemeral additions.

**How Hermes does it** (`agent/system_prompt.py`, `agent/prompt_builder.py`):

The cached system prompt is three tiers, joined in this order:

| Tier | Contents |
|---|---|
| **stable** | Identity (`SOUL.md` or a default identity), tool-usage guidance, model-family-specific guidance, platform hint (e.g. "you're in a terminal, avoid Markdown") |
| **context** | Caller-supplied system message, project context files (`AGENTS.md` etc.), git workspace snapshot, operator instructions |
| **volatile** | Skills index, frozen `MEMORY.md` snapshot, frozen `USER.md` snapshot, external memory provider block, timestamp/session/model line, runtime environment (host, home, cwd) |

Key details:

- **Ordering is chosen for cache sharing.** Shared project files come *before* anything that
  names the current worktree/cwd, so two sessions of the same project in different git
  worktrees still share a long common prefix (which longest-prefix caches reuse).
- **Guidance is conditional on the tool list.** E.g. skill-writing instructions only appear if
  `skill_manage` is available; if memory is disabled, the memory guidance block disappears too,
  so the model is never told about a tool it cannot call.
- **Model-family-specific nudges.** For GPT/Codex models a "tool-use enforcement" block is
  added ("You MUST use your tools to take action — do not describe what you would do").
- **A strong default identity** that sets response style: match reply length to the weight of
  the ask, no filler, no restating the request, no narrating visible tool calls, "agree because
  it's right, not because the user said it".
- **Ephemeral layers never enter the cached prompt:** an ephemeral system prompt, prefill
  messages, gateway context overlays, per-turn memory recall, and plugin `pre_llm_call` context
  are all appended to the **current user message** at API-call time.
- **Change notes instead of rebuilds.** If the skill index changes mid-session (a skill was
  installed in another terminal), the next user message carries a one-shot note listing the
  added/removed skills. Same for switching surfaces (desktop ↔ TUI): the new platform guidance
  arrives as a one-shot note. The system prompt itself converges at the next compaction.

**Why it works:** the expensive, rarely changing content is cached once; everything dynamic
rides in the cheap, uncached tail of the request.

**Copy it:**
1. Build the system prompt from ordered sections: stable → project → volatile.
2. Freeze it at session start; store it with the session so a resumed session sends the same
   bytes.
3. Add a "turn context" channel that appends dynamic notes to the latest user message.
4. Compute "what changed since the prompt was built" and send it once as a note.

**Source:** `agent/system_prompt.py`, `agent/prompt_builder.py`, `agent/skills_index_delta.py`,
`agent/surface_switch.py`, `website/docs/developer-guide/prompt-assembly.md`.

---

## 3. Bounded, curated persistent memory

**What it is:** two small Markdown files the agent maintains itself and that are injected into
every new session.

**How Hermes does it** (`tools/memory_tool.py`, `tools/memory_tool_store.py`):

| File | Purpose | Default limit |
|---|---|---|
| `MEMORY.md` | The agent's notes about the **environment**: OS, tools, project conventions, quirks, paths, lessons | 2,200 chars (~800 tokens) |
| `USER.md` | A profile of the **user**: name, role, timezone, preferences, communication style, pet peeves | 1,375 chars (~500 tokens) |

Mechanics worth copying exactly:

- **Hard character limits, no auto-compaction.** When a write would overflow, the tool returns
  an error that includes the current entries, current usage, and *exactly how many characters
  must be freed*. The agent then sends one batch that removes/shortens entries **and** adds the
  new one; the limit is checked on the batch's final result. This forces the agent to curate
  instead of hoard.
- **The usage gauge is in the prompt.** The rendered block has a header like
  `MEMORY (your personal notes) [67% — 1,474/2,200 chars]`, so the model knows when to
  consolidate. Entries are separated by `§`.
- **Frozen snapshot.** The snapshot is captured at session start and never changes mid-session
  (cache!). Writes go to disk immediately and show up in the *next* session.
- **No `read` action.** Memory is already in the system prompt, so a read tool would only waste
  calls.
- **Edit semantics designed against model mistakes:**
  - `add`, `patch` (replace one *span* inside an entry), `remove`.
  - `old_text` is a short **unique substring**; ambiguous matches error out and ask for more
    context.
  - A former `replace` action was **removed** because models used it as find-and-swap and wiped
    the other facts in multi-fact entries. It now returns an error pointing at `patch`.
  - A failed match returns the `closest_entries` so the model can self-correct.
- **Required `target`** (`memory` or `user`) at the top level of every call; mismatches are
  rejected before writing.
- **Exact duplicates** are rejected (returned as success with "no duplicate added").
- **Security scan on write.** Entries are scanned for prompt injection, exfiltration patterns
  and invisible Unicode — memory is injected into the system prompt, so it is an attack surface.
- **Optional write approval.** `memory.write_approval: true` makes foreground writes prompt
  inline in the CLI and *stages* all other writes (background review, messaging) for
  `/memory pending | approve <id> | reject <id>`. A staged patch/remove pins the exact entry it
  targets; if that entry changed since staging, the approval is refused.
- **Guidance on what to save vs skip** lives in the prompt: save preferences, environment facts,
  corrections, conventions, explicit "remember" requests; skip trivia, things easily
  re-discovered, raw dumps, session ephemera, and anything already in context files.
- **Memory needs session boundaries.** The docs explicitly teach users to run `/new` at natural
  boundaries — memory and session search only pay off when sessions end.

**Why it works:** a tiny, always-present memory is cheap and reliably used; the hard limit plus
an informative error turns the model into an editor of its own memory rather than a logger.

**Copy it:** two Markdown files, a single `memory` tool with `add/patch/remove` + `target`,
character limits with an error that tells the model how much to free, a usage header in the
prompt, frozen-at-session-start injection, and an injection scan on write.

**Source:** `tools/memory_tool.py`, `tools/memory_tool_store.py`, `agent/memory_manager.py`,
`website/docs/user-guide/features/memory.md`.

---

## 4. Session store + session search

**What it is:** every conversation is stored in SQLite with full-text search, and the agent has
a tool to search and scroll through past sessions.

**How Hermes does it** (`hermes_state*.py`, `tools/session_search_tool.py`):

- **Schema:** a `sessions` table (id, title, source, parent session, model, token usage, …) and
  a `messages` table; an FTS5 virtual table indexes `content`, `tool_name` and `tool_calls`,
  kept in sync by INSERT/UPDATE/DELETE triggers.
- **One tool, four shapes, no `mode` parameter** — the shape is inferred from which arguments
  are set:
  1. **Discovery** (`query`): FTS5 search, results deduped by session lineage. Adaptive detail:
     the top hit returns its full context (±5 messages around the match plus "bookends" — the
     first and last 3 user/assistant messages, which reconstruct goal → match → resolution);
     lower hits stay compact.
  2. **Scroll** (`session_id` + `around_message_id`): a ±`window` slice. Scroll forward/back by
     passing the first/last message id back. ~1–2 ms.
  3. **Read** (`session_id` only): whole session, or a bounded head/tail view if large.
  4. **Browse** (no args): recent sessions.
- **No LLM calls** in the search path — tens of milliseconds, zero token cost until used.
- **Prompt guidance:** "When the user references a past conversation, use `session_search`
  before asking them to repeat themselves."
- **Session lineage:** compression can create a child session linked by `parent_session_id`, so
  history remains navigable across compactions.
- **Session titles** are generated in two stages off the critical path: an instant
  deterministic title, then an upgrade from one cheap model call. Provenance ordering
  `derived < llm < user` means a title the user typed is never overwritten.
- **Multi-process write safety:** short SQLite timeout (1 s) plus application-level retry with
  random jitter (to avoid the "convoy effect" of SQLite's deterministic backoff), `BEGIN
  IMMEDIATE` transactions, WAL mode with periodic passive checkpoints, and separate retry
  budgets for transcript writes (60 s) vs routine (20 s) vs observational writes (0.5 s).

**Why it works:** memory holds the few facts needed *every* session; session search covers the
long tail ("what did we decide about X last week?") at zero standing cost.

| | Persistent memory | Session search |
|---|---|---|
| Capacity | ~1,300 tokens | Unlimited |
| Cost | Every prompt | Only when called |
| Curated by | The agent | Automatic |
| Use | Key facts always in context | Finding specifics from past chats |

**Copy it:** SQLite + FTS5, one search tool with discovery/scroll/read/browse shapes, bookends
on the top hit, and a prompt line telling the model to search before asking.

**Source:** `hermes_state_schema.py`, `hermes_state_fts.py`, `hermes_state_search.py`,
`hermes_state_titles.py`, `tools/session_search_tool.py`, `agent/title_generator.py`,
`website/docs/developer-guide/session-storage.md`.

---

## 5. Skills: procedural memory with progressive disclosure

**What it is:** reusable "how to do this class of task" documents that the agent loads only
when relevant, and that it can write and improve itself.

**How Hermes does it** (`tools/skills_tool.py`, `tools/skill_manager_tool.py`):

**Format** — a directory per skill with a `SKILL.md` (YAML frontmatter + Markdown body) and
optional support files:

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
platforms: [macos, linux]         # optional OS restriction
metadata:
  hermes:
    tags: [python, automation]
    category: devops
    requires_toolsets: [terminal]  # only show when terminal exists
    fallback_for_toolsets: [web]   # only show when web is MISSING
---
# Skill Title
## When to Use
## Procedure
## Pitfalls
## Verification
```

```
my-skill/
├── SKILL.md         # always-on rules + procedure
├── references/      # topical depth needed only sometimes
├── templates/       # starter files to copy and modify
└── scripts/         # re-runnable deterministic actions
```

**Progressive disclosure** — three loading levels:

```
Level 0: skills index in system prompt / skills_list()  → name + one-line description
Level 1: skill_view(name)                               → full SKILL.md
Level 2: skill_view(name, file_path)                    → a specific support file
```

The system prompt carries a compact `<available_skills>` index grouped by category with the
instruction: *"Before replying, scan the skills below. If one clearly matches your task, load
it with `skill_view(name)` and follow its instructions."*

**Conditional activation** — `requires_tools(ets)` / `fallback_for_tool(set)s` let a skill
appear only when it's useful (e.g. a free web-scraping fallback that shows only when the paid
web tool is unavailable). `platforms` hides OS-specific skills elsewhere.

**Agent-managed skills** — the `skill_manage` tool:

| Action | Use |
|---|---|
| `create` | New skill (full `SKILL.md`) |
| `patch` | Targeted `old_string` → `new_string` (preferred; token-efficient) |
| `patch` + `content` | Full rewrite |
| `delete` | Remove |
| `write_file` / `remove_file` | Support files under `references/`, `templates/`, `scripts/` |

Guard rails:

- **Read-before-write is enforced:** patching an existing skill requires a fresh `skill_view`
  in the same run; a transcript quote doesn't count. Prevents blind overwrites from stale
  memory.
- **Each action's text slot is exclusive** (`content` → create, `new_string` → patch,
  `file_content` → write_file); a mismatched slot rejects the whole batch before any op applies.
- **Advisory linter** on writes, with rules aimed at known failure shapes:
  `incident-log-shape` (body full of PR/issue numbers), `references-sprawl` (> 60 reference
  files), `oversized-body` (> ~24k chars — it stays in context the rest of the session). Warns,
  never blocks.
- **Pinning and locking:** pinned skills can't be deleted or auto-archived; locked skills can't
  be changed by the agent at all.
- **Optional write approval** for skills, mirroring memory.
- **Provenance + audit ledger:** every mutation is recorded with the actor (`agent`, `curator`,
  `user`) and supports single-edit rollback.

**Skills as slash commands** — every skill is invokable as `/<skill-name> <task>`; several can
be stacked; **bundles** alias a set of skills under one name.

**`/learn`** — build a skill from sources the user points to (directories, URLs, this chat,
notes). Large sources become knowledge-base skills with `references/`.

**Compression awareness** — when compaction removes a loaded skill's text, the result is
replaced with `[SKILL_PRUNED: content lost in compression; reload with skill_view(name='…')]`,
the summary lists pruned skills, and a "Skill Safety Rule" tells the model to reload before
relying on it. Skills preloaded into the system prompt are never compacted.

**Installing third-party skills** — a hub with a static security scan (regex + AST audit) and
trust levels: builtin (never scanned), trusted sources (warnings allowed), community (any
finding blocks unless `--force`). Project-local skills require explicitly trusting the project.

**Why it works:** the context only pays for a one-line description per skill until a skill is
actually needed. Skills are where *procedures* and *user preferences about how to do a task*
live; memory stays tiny.

**Copy it:** a `skills/` directory of `SKILL.md` files, an index in the prompt, `skills_list` +
`skill_view(name, file_path?)` tools, a `skill_manage` tool with patch-first semantics and
read-before-write enforcement, and the compression marker.

**Source:** `tools/skills_tool.py`, `tools/skill_manager_tool.py`, `tools/skill_manager_guards.py`,
`tools/skill_linter.py`, `tools/skills_guard.py`, `tools/skills_ast_audit.py`,
`tools/skill_provenance.py`, `tools/skill_ledger.py`, `agent/skill_commands.py`,
`agent/skill_bundles.py`, `agent/learn_prompt.py`, `website/docs/user-guide/features/skills.md`.

---

## 6. The background self-improvement review

**This is the heart of "an agent that learns".** After turns, a forked copy of the agent reviews
the conversation and decides what to save to memory and which skills to create or patch —
without touching the live conversation.

**How Hermes does it** (`agent/background_review.py`):

- **Triggers:** counters on the agent. `memory.nudge_interval` (default 10 user turns) and
  `skills.creation_nudge_interval` (default 10 tool iterations). When a counter crosses its
  threshold, a review is scheduled after the turn completes and the counter resets. Users can
  also force one with `/refine`.
- **The fork:** a daemon thread spawns a forked `AIAgent` that:
  - inherits the parent's provider, model, credentials **and cached system prompt**, so it hits
    the same prefix cache (the review is cheap);
  - replays a snapshot of the conversation, then gets one review prompt (memory-only,
    skills-only, or combined);
  - runs under a **dispatch-side tool whitelist** (memory + skill tools only);
  - has its own nudge counters set to 0 (no recursive reviews) and compression detached;
  - is cancelled if the user starts a new live turn (`cancel_background_review_for_live_turn`);
  - attributes its token usage back to the parent session.
- **The review prompts are the real asset.** Highlights (paraphrased; full text in the file):

  *Memory routing:*
  - Two stores: `USER.md` = who the user is; `MEMORY.md` = facts about the environment.
  - **One fact goes to ONE store, never both.**
  - **Save only facts grounded in this conversation** — a user message or tool result you can
    point to. Never infer traits, skill level, role, mood or preferences from topic/tone/tools.
    Never turn a one-off request into a standing preference unless the user marked it general
    ("always", "from now on", "remember") or stated a standing like/dislike. *If unsure, leave
    it out — a wrong fact misleads every future session.*

  *What a skill is:* the instructions for doing a **class** of task the most efficient and
  correct way, to this user's specifications.
  - Procedure first, in order, with concrete commands; pitfalls attach to the step they affect.
  - A pitfall is a **generalizable rule + one clause of WHY**, in the imperative — not a
    narrative of what happened.
  - No PR numbers, dates, ticket IDs, or quoted chat as content.
  - The same lesson twice is ONE rule — search before adding.
  - Don't duplicate what's already always loaded (AGENTS.md, tool schemas).
  - Fix wrong text in place; never append "UPDATE: actually…".

  *Signals that warrant a skill update:* user corrected style/format/verbosity (frustration is
  a **first-class skill signal**, not just a memory signal); user corrected the workflow; a
  non-trivial technique emerged; a loaded skill turned out wrong or outdated.

  *Preference order:* (1) patch a skill that was loaded this session → (2) patch an existing
  class-level umbrella → (3) add a support file under an umbrella → (4) create a new
  class-level skill. Names must be class-level, never "fix-X-today".

  *Do NOT capture:* environment-dependent failures (missing binaries, unconfigured creds);
  **negative claims about tools** ("browser tools don't work" — these harden into self-imposed
  refusals long after the problem is fixed); transient errors; one-off task narratives;
  **unresolved failures dressed up as a "recommended approach"**. If a tool failed due to
  setup, capture the *fix* under a setup/troubleshooting skill instead.

  *Bias to action:* "Nothing to save" is allowed but should not be the default; most sessions
  produce at least one small skill update.

- **Preference placement rule:** a user preference about how to do a task lives in the skill
  that governs that task; `USER.md` only holds cross-cutting preferences no skill owns — never
  both.
- **Visibility:** the parent shows a short summary of what the review saved; with write
  approval on, everything is staged for human review.

**Why it works:** learning happens *after* the user got their answer, in a cache-warm fork, with
prompts that have been tuned against concrete failure modes (hoarding, misrouting, inferring
traits, hardening negative claims).

**Copy it:** turn counters → post-turn background fork with the same cached prefix and a tool
whitelist → one review prompt. Start by copying the prompt text in
`agent/background_review.py` (`_MEMORY_ROUTING_BLOCK`, `_LESSON_LAYER_BLOCK`,
`_DO_NOT_CAPTURE_BLOCK`, `_SKILL_REVIEW_PROMPT`, `_COMBINED_REVIEW_PROMPT`) — it encodes a lot
of hard-won lessons.

**Source:** `agent/background_review.py`, `agent/review_engine.py`, `agent/agent_init.py`
(nudge counters), `agent/learning_graph.py` (the `/journey` timeline of what was learned).

---

## 7. The curator: keeping the skill library healthy

**What it is:** a periodic background maintenance pass so self-written skills don't pile up
forever.

**How Hermes does it** (`agent/curator.py`):

- **Triggered by inactivity, not a timer:** runs when `interval_hours` (default 7 days) have
  passed **and** the agent has been idle `min_idle_hours` (default 2). The first observation on
  a fresh install only seeds the timestamp (no surprise run on day one).
- **Phase 1 — deterministic transitions (no LLM):** usage is tracked per skill (views, uses,
  patches). Unused 14 days → `stale`; unused 30 days → moved to `.archive/` (restorable).
  - Pinned skills and skills referenced by any cron job (even paused) are exempt.
  - Never-used skills get a grace floor — *"zero uses is absence of evidence, not proof the
    skill is disposable."*
  - By default only **agent-created** skills are managed; bundled/hub skills are opt-in.
- **Phase 2 — LLM consolidation (opt-in):** a forked agent surveys skills and merges
  overlapping ones into class-level umbrellas. Off by default because it's expensive.
- **Safety:** snapshot backups before runs, `rollback`, a per-mutation audit ledger with
  single-edit rollback, `--dry-run`, pin/lock/adopt commands, archive TTL purge.

**Copy it:** track `use_count` + `last_used_at` per skill; run a cheap deterministic
stale/archive pass when idle; archive, don't delete; make LLM consolidation optional.

**Source:** `agent/curator.py`, `agent/curator_backup.py`, `tools/skill_usage.py`,
`website/docs/user-guide/features/curator.md`.

---

## 8. Pluggable memory providers

**What it is:** an interface so external memory systems (vector stores, user-modeling services)
plug in alongside or instead of the built-in files.

**How Hermes does it** (`agent/memory_provider.py`, `agent/memory_manager.py`):

The `MemoryProvider` ABC defines a clean lifecycle — this list is worth copying as-is:

| Hook | When | Use |
|---|---|---|
| `is_available()` | Agent init | Config check — **no network calls** |
| `initialize(session_id, …)` | Startup | Bind to session/profile |
| `get_tool_schemas()` / `handle_tool_call()` | Tool injection / use | Provider-specific tools |
| `system_prompt_block()` | Prompt assembly | Static info (enters the cached prompt) |
| `prefetch(query)` | Before each API call | Return recalled context (goes into the **user message**, not the system prompt) |
| `queue_prefetch(query)` | After each turn | Pre-warm the next turn's recall |
| `sync_turn(user, assistant, messages)` | After each turn | Persist the exchange (on a background thread) |
| `on_pre_compress(messages)` | Before compression | Extract insights before detail is lost |
| `on_memory_write(action, target, content)` | Built-in memory writes | Mirror to your backend |
| `on_session_end(messages)` | Session end | Final extraction/flush |
| `shutdown()` | Process exit | Cleanup |

Notable rules:

- **Single active provider** at a time (plus the built-in store).
- **Oversized recall is capped** (spilled to a file above a safety ceiling) so a runaway
  provider can't flood the context.
- **Fail-closed pre-compress checkpoints (opt-in):** a provider can declare it durably
  archives transcripts; with `compression.checkpoint_required: true`, compression refuses to
  run unless that checkpoint succeeds. Checkpoints must be idempotent (key by content digest).
- **Profile isolation:** providers bind state on the instance at `initialize`, never globals.
- `nudge_interval` defaults to 0 when an external provider auto-extracts, to avoid
  double-saving.

**Copy it:** the hook table above. `prefetch`→user-message and `sync_turn`→background-thread
are the two that matter most.

**Source:** `agent/memory_provider.py`, `agent/memory_manager.py`, `plugins/memory/`,
`website/docs/developer-guide/memory-provider-plugin.md`.

---

## 9. Context compression

**What it is:** when the conversation approaches the context limit, summarize the middle and
keep the beginning and end.

**How Hermes does it** (`agent/context_compressor*.py`):

**Triggers:**
- In-loop compressor at **50%** of the context window (configurable, per-model overrides).
- A gateway "hygiene" safety net at **85%** before a message is processed (catches sessions that
  grew between turns, e.g. overnight). It's deliberately higher — at 50% it fired every turn.
- Manual `/compress` (optionally `here [N]` to keep the last N turns, `--preview` to dry-run).

**Token accounting:** prefer the provider's reported usage from the last response plus a rough
estimate of only the newly appended messages ("usage anchor"); fall back to a char-based
estimate only when needed. A rough estimate over threshold waits one request for real usage
rather than compressing prematurely. Image token cost is *learned* from real usage per model.

**The 4-phase algorithm:**

1. **Prune old tool results (no LLM):** tool outputs > 200 chars outside the protected tail are
   replaced with `[Old tool output cleared to save context space]`. Cheap and very effective.
   A tool round the model hasn't answered yet stays verbatim.
2. **Determine boundaries:** protect the head (system + first exchange) and a tail sized by
   **token budget** (walk backward accumulating tokens; capped at 20% of the window), with a
   minimum message count. Boundaries are aligned so a `tool_call` and its `tool_result`s are
   never split.
3. **Structured summary** of the middle, via an auxiliary model, using a fixed template:
   ```
   ## Goal
   ## Constraints & Preferences
   ## Progress
   ### Done / ### In Progress / ### Blocked
   ## Key Decisions
   ## Relevant Files
   ## Next Steps
   ## Critical Context   (exact values, error messages, config)
   ```
   Summary budget = 20% of compressed content, min 2,000 tokens, max min(5% of window, 12,000).
4. **Assemble:** head + summary (role chosen to keep valid user/assistant alternation) + tail.
   Orphaned tool pairs are repaired: results without calls removed; calls without results get
   a stub result.

**Iterative re-compression:** the next compaction passes the previous summary and asks the
model to **update** it (move items from In Progress to Done, drop obsolete info) rather than
summarize a summary from scratch.

**Robustness:**
- Memory is flushed and memory providers' `on_pre_compress` runs **before** the lossy rewrite.
- Failure cooldowns escalate (60 s → 300 s → 900 s) so a broken summary backend doesn't retry
  every turn; a deterministic fallback summary is committed if the summary call fails.
- The todo list is re-injected after compression (see §23).
- Read-dedup state for files is reset so the model can re-read a file once in full (see §16).

**Optional variants:**
- **Prepared compaction:** start writing the next summary in the background when within 12% of
  the trigger, so compaction is instant when it fires.
- **Micro-compaction:** after each turn, fold the oldest un-absorbed exchange into a rolling
  summary — pays the cost in installments, but breaks the cache prefix every turn (off by
  default; documented trade-off).
- **Pluggable `ContextEngine`** ABC (`should_compress()`, `compress()`, optional tools) so a
  plugin can replace lossy summarization (e.g. with lossless context management).

**Removed on purpose:** intermediate "context pressure" warnings to the model — they made
models give up prematurely on complex tasks.

**Copy it:** phase 1 (prune old tool outputs) + tail-by-token-budget + tool-pair-safe
boundaries + the summary template + iterative update. That's 90% of the value.

**Source:** `agent/context_compressor.py` and `context_compressor_*.py`,
`agent/conversation_compression*.py`, `agent/prepared_compaction.py`, `agent/micro_compaction.py`,
`agent/context_engine.py`, `website/docs/developer-guide/context-compression-and-caching.md`.

---

## 10. Prompt caching strategy

**How Hermes does it** (`agent/prompt_caching.py`):

- **"system_and_3"** for Anthropic's 4 breakpoints: one on the system prompt, plus a rolling
  window on the last 3 non-system messages. After compression the system-prompt cache survives
  and the rolling window re-establishes within 1–2 turns.
- **TTL by pace:** `"auto"` picks 1 h for human-paced sessions (people take breaks) and 5 min
  for machine-paced sources (subagents, cron, webhooks, API).
- **Rules the codebase enforces:** never mutate past messages; append-only history; compression
  only appends a note to the system prompt on the *first* compaction; model identity and
  account are part of the cache key — never silently swap the model or credentials mid-session
  (user-facing docs for `/model`, fallback and key rotation carry cost warnings).
- **Forks reuse the parent's cache:** background review, `/btw` side questions and the
  curator replay the parent's exact prefix so they're cheap.

**Copy it:** the rolling-window breakpoint strategy and the append-only discipline.

**Source:** `agent/prompt_caching.py`, `agent/prompt_cache_boundary.py`,
`agent/prompt_cache_scope.py`.

---

## 11. Context files, subdirectory hints and references

**Project context files** (`agent/prompt_builder.py`):
- Priority, first match wins: `.hermes.md`/`HERMES.md` (walks up to git root) → `AGENTS.md` →
  `CLAUDE.md` → `.cursorrules` / `.cursor/rules/*.mdc`. Compatible with files written for other
  agents.
- A global `SOUL.md` in the agent home replaces the default identity (persona).
- **Scanned** for prompt injection (invisible Unicode, "ignore previous instructions",
  exfiltration); a project file with a hit is replaced by `[BLOCKED: …]`. The user's own
  `SOUL.md` is warned about but loaded (it's user-authored).
- **Truncated** with a 70% head / 20% tail split around a marker; the cap scales with the
  model's context window (20k-char floor, 500k ceiling).

**Progressive subdirectory hints** (`agent/subdirectory_hints.py`): as the agent navigates into
subdirectories via tool calls, any `AGENTS.md`/`CLAUDE.md`/`.cursorrules` found there is
appended **to the tool result** — so nested instructions arrive exactly when relevant, without
touching the system prompt. This is an excellent, cheap pattern.

**@-references** (`agent/context_references.py`): users can write `@file:path`, `@folder:path`,
`@diff`, `@git:…`, `@url:…` (plus plugin prefixes) in a message to inline that content.

**`/init`:** generate or update `AGENTS.md` from a repo scan.

**Source:** `agent/prompt_builder.py`, `agent/context_file_sources.py`,
`agent/subdirectory_hints.py`, `agent/context_references.py`,
`website/docs/user-guide/features/context-files.md`.

---

## 12. Tool registry, toolsets and availability checks

**How Hermes does it** (`tools/registry.py`, `model_tools.py`, `toolsets.py`):

- **Self-registering tools:** each module calls `registry.register(name, toolset, schema,
  handler, check_fn, requires_env, is_async, emoji)` at import time. Discovery **AST-scans**
  `tools/*.py` for top-level `registry.register()` calls and imports only those modules — no
  manual list, and helper modules aren't imported.
- **Name collisions are rejected** across toolsets unless explicitly overridden.
- **`check_fn` availability gating:** each tool can declare a function (API key present?
  dependency installed? service configured?). Unavailable tools are simply **not sent** to the
  model. Exceptions count as unavailable (fail-safe). Results are cached per build.
- **Toolsets:** named bundles (`web`, `file`, `terminal`, `browser`, `memory`, …), composable
  (`includes=[…]`), plus presets per surface (`hermes-cli`, `hermes-telegram`, `safe`,
  `coding`, …). Users enable/disable by toolset.
- **Dynamic schema patching:** after filtering, schemas that *mention other tools*
  (`execute_code`, `delegate_task`, …) are rewritten to list only tools that actually passed
  filtering — preventing the model from hallucinating calls to unavailable tools.
- **Two-level error wrapping:** handler exceptions become `{"error": "Tool execution failed:
  Type: msg"}`; the model always gets well-formed JSON, never a crash.
- **Agent-level tools** (`todo_list`, `memory`, `session_search`, `delegate_task`) are
  intercepted in the loop because they need agent state; their registry handlers return an
  error if reached directly.
- **Async bridging:** persistent event loop for CLI, disposable thread for a running loop,
  per-thread loops for parallel tool workers.
- **Parallel tool execution:** multiple tool calls run in a thread pool; interactive tools
  (`clarify`) force sequential; results are reinserted in original order; batches touching
  overlapping file paths are not parallelized.
- **Argument coercion** (`tools/arg_coercion.py`) and **schema sanitizing** per provider
  (`tools/schema_sanitizer.py`) make sloppy model arguments land.

**Copy it:** decorator/registration with `check_fn`, toolset bundles, and dynamic schema
patching. These keep the tool list honest and small.

**Source:** `tools/registry.py`, `model_tools.py`, `toolsets.py`,
`website/docs/developer-guide/tools-runtime.md`, `website/docs/developer-guide/adding-tools.md`.

---

## 13. Tool Search: deferring rarely used tools

**What it is:** when there are many tools (e.g. dozens of MCP tools), don't send them all —
send three meta-tools instead.

**How Hermes does it** (`tools/tool_search*.py`):

```
tool_search(queries, limit?)   search the deferred-tool catalog (several queries at once)
tool_describe(names)           load full schemas for chosen tools
tool_call(calls)               invoke deferred tools: [{name, arguments}, …] (up to 10)
```

Each entry in a `tool_call` batch is split into its own normal tool call internally, so it still
gets scope checks, schema validation, hooks, approval and its own result. Tools can be marked
deferred by default (e.g. the browser credential vault).

**Why it works:** the always-sent tool list stays small and stable (cache-friendly), while the
long tail of tools stays reachable.

**Source:** `tools/tool_search.py`, `tools/tool_search_catalog.py`,
`tools/tool_search_validation.py`, `website/docs/user-guide/features/tool-search.md`.

---

## 14. Programmatic tool calling (`execute_code`)

**What it is:** the model writes a Python script that calls tools as functions; only the
script's printed output returns to the model.

**How Hermes does it** (`tools/code_execution_tool.py`, `tools/code_execution_rpc.py`):

1. Generate a `hermes_tools.py` stub module whose functions are RPC calls.
2. Open a Unix domain socket and an RPC listener thread.
3. Run the script in a child process; each tool call travels over the socket back to the
   harness, which dispatches it normally.
4. Return only `print()` output — intermediate results never enter the context.

```python
from hermes_tools import web_search, web_extract
results = web_search("Python 3.13 features", limit=5)
for r in results["data"]["web"]:
    content = web_extract([r["url"]])
    # filter and process...
print(summary)
```

- Only a safe subset of tools is exposed (web, file read/write/search/patch, foreground
  terminal).
- A **persistent session kernel** keeps variables between calls.
- Resource limits on time and output.
- Prompt guidance: use it when there are **3+ tool calls with logic between them**, loops,
  bulk filtering, or branching.

**Why it works:** collapses multi-step pipelines into one model turn and keeps bulky
intermediate data out of the context window.

**Source:** `tools/code_execution_tool.py`, `tools/code_execution_rpc.py`,
`tools/code_kernel.py`, `website/docs/user-guide/features/code-execution.md`.

---

## 15. Subagent delegation

**What it is:** spawn child agents with fresh, isolated context for subtasks, alone or in
parallel.

**How Hermes does it** (`tools/delegate_tool*.py`, `agent/subagent_lifecycle.py`):

- **Fresh context by design.** "Subagents know nothing" — the child sees only the `goal` and
  `context` the parent writes (plus the project's context files). The tool description teaches
  the parent to pass everything (file paths, error text, versions).
- **Structured return:** the child is told to report what it did, what it found, files modified
  and issues encountered. Optional `output_schema` forces JSON output.
- **Parallel batches** (default max 10 concurrent; a batch over the limit is an error, never
  silently truncated); results returned in order.
- **Tool inheritance, never escalation:** there's no model-facing `toolsets` parameter; children
  inherit the parent's tools, so the model can't grant a child more power than it has.
- **Blocked in children:** `delegate_task` (unless `role="orchestrator"`, bounded by
  `max_spawn_depth`), `clarify` (can't talk to the user), `memory` (no writes to shared memory),
  `send_message` (no side effects on platforms), `cronjob` (no scheduling). `execute_code` is
  kept.
- **Own iteration budget** per child (configurable), independent of the parent's.
- **Background mode** with completion delivered later as a notification; optional **git
  worktree** per child so parallel edits don't collide; per-child model/provider routing (use a
  cheaper model for children); image forwarding for visual tasks.
- **Cross-agent file-state coordination:** a process-wide registry tracks per-agent read stamps
  and the last writer per path, so subagent A can't clobber a file subagent B wrote after A
  read it (see §16).
- **Prompt cache TTL** is 5 min for subagents (machine-paced).

**Copy it:** isolated child with goal+context, inherited-not-granted tools, a blocklist
(`clarify`, `memory`, recursive delegate), a structured summary contract, bounded concurrency.

**Source:** `tools/delegate_tool.py` and `delegate_tool_*.py`, `tools/async_delegation*.py`,
`tools/subagent_worktree.py`, `tools/file_state.py`, `agent/subagent_lifecycle.py`,
`website/docs/user-guide/features/delegation.md`.

---

## 16. File-editing safety

Models make predictable editing mistakes. Hermes has a guard for most of them.

**Read tracking** (`tools/file_tools_read_tracking.py`), per task:
- **Loop detection:** repeated identical reads/searches are counted (reset by any other tool
  call) and a warning is returned when the model re-reads the same thing in a loop.
- **Dedup:** re-reading an unchanged file returns a short stub instead of the full content —
  but the dedup set is cleared at compaction so one recovery read gets full content again.
- **Staleness warnings:** read timestamps let the tool warn if a file changed on disk since it
  was read.
- **Read coverage:** tracks which line ranges have been paged through at a given file version.

**Write guards** (`tools/file_tools_write_guards.py`), applied in order:
1. Sensitive path → hard deny.
2. Binary document write → refuse.
3. Protected instruction files (e.g. agent config/context files) → always ask.
4. Approval-required paths → normal approval gate.
5. Cross-profile paths → refuse (lost-work protection).
6. **Stale-overwrite blocker:** a whole-file `write_file` over content this task never read, or
   that changed since it read it, is refused (under a per-path lock).

**Fuzzy patching** (`tools/fuzzy_match.py`): `patch` uses an ordered chain of increasingly
permissive match strategies so whitespace/indentation/escaping/Unicode drift in the model's
`old_string` still lands on the intended region. **Escape-drift autocorrect** halves
doubled backslashes only when the corrected anchor appears verbatim in the file, and refuses
ambiguous cases. V4A multi-file patch format (`*** Begin Patch`) is also supported.

**Post-edit lint** (`tools/file_operations_lint.py`) reports syntax errors immediately after
an edit.

**Self-repo guard:** refuses edits to the harness's own installation by accident.

**Honest framing:** `agent/file_safety.py` states these are *defense-in-depth, not a security
boundary* — the terminal can still write anything. The value is clear denials for models that
respect tool errors, and an audit trail.

**Copy it:** read-before-overwrite, stale-file refusal, fuzzy `old_string` matching, and
post-edit lint. These eliminate the most common "the agent destroyed my file" incidents.

**Source:** `tools/file_tools.py`, `tools/file_tools_read_tracking.py`,
`tools/file_tools_write_guards.py`, `tools/file_state.py`, `tools/fuzzy_match.py`,
`tools/escape_drift_autocorrect.py`, `tools/patch_parser.py`, `tools/file_operations_lint.py`.

---

## 17. Tool output limits and spill-to-file

**How Hermes does it:**

- **Per-tool caps** (`tool_output.max_bytes` for terminal, `max_lines` for read pagination,
  `max_line_length` per line).
- **One truncation algorithm everywhere** (`tools/tool_output_truncate.py`): **40% head** (errors
  surface early) / **60% tail** (most recent output matters most) around a single marker
  `... [<LABEL> TRUNCATED - N <unit> omitted out of T total] ...`.
- **Spill instead of losing data** (`tools/tool_result_storage.py`): output over a threshold is
  written to `$HOME/cache/spillover/{id}.txt` and replaced with a preview + path. The model can
  read the file if it needs the middle. Remote backends get the in-sandbox path.
- **Per-turn budget** across all tool results in a turn.
- **Old tool outputs are pruned at compaction** (§9).

**Copy it:** head/tail truncation with a single, recognizable marker + spill the full output
to a file the model can read later.

**Source:** `tools/tool_output_limits.py`, `tools/tool_output_truncate.py`,
`tools/tool_result_storage.py`, `tools/hook_output_spill.py`.

---

## 18. Filesystem checkpoints (shadow git)

**What it is:** automatic, invisible snapshots of the working directory before the agent
mutates files, with one-command rollback.

**How Hermes does it** (`tools/checkpoint_manager.py`):
- Snapshot **once per turn** before the first file-mutating operation (`write_file`, `patch`,
  destructive terminal commands).
- Stored in a **single shared shadow git store** under the agent's home (objects deduplicated
  across projects) — the user's own repo is never touched.
- **Not a tool:** the model never sees it; it's pure infrastructure.
- `/rollback` lists and restores checkpoints; restores keep the user's own hand-edits unless
  `--all`. Pruning keeps the store bounded.

**Copy it:** `git --git-dir=<shadow> --work-tree=<project> add -A && commit` before the first
mutation per turn. Cheap insurance that makes users trust the agent with edits.

**Source:** `tools/checkpoint_manager.py`, `tools/checkpoint_pruning.py`.

---

## 19. Dangerous-command approval

**How Hermes does it** (`tools/approval*.py`):

- **Pattern detection:** `DANGEROUS_PATTERNS` is a list of `(regex, description)` covering
  recursive deletes, `mkfs`/`dd`, destructive SQL (`DROP TABLE`, `DELETE` without `WHERE`),
  writes to `/etc`, service manipulation, `curl | sh`, fork bombs, process kills, etc.
- **Approval routing per surface:** CLI shows an inline prompt (approve / deny / always); a
  messaging platform gets an async approval request (`/approve`, `/deny [reason]`); batched
  approvals for multiple commands.
- **Smart approval (optional):** an auxiliary model can auto-approve low-risk matches (e.g.
  `rm -rf node_modules/` matches "recursive delete" but is harmless).
- **Session memory of approvals:** approving a category once covers the rest of the session.
- **Permanent allowlist:** "always" writes the pattern to config.
- **Approval floors:** some categories can never be auto-approved.
- **YOLO mode** (`/yolo`) for users who opt out explicitly.
- Denial reasons are returned to the model so it can adapt.

**Copy it:** regex patterns + an approval callback interface implemented per UI + per-session
"approved categories" + a persistent allowlist. Add an LLM-judge tier only once the basics work.

**Source:** `tools/approval.py`, `tools/approval_detection.py`, `tools/approval_smart.py`,
`tools/approval_floors.py`, `tools/approval_prompt.py`, `tools/write_approval.py`.

---

## 20. Prompt-injection and secret hygiene

- **One shared threat-pattern library** (`tools/threat_patterns.py`) used for context files,
  memory writes, skill installs and tool results. Each pattern has a **scope**: `all`,
  `context` (adds role-hijack / C2 patterns for untrusted content — warn-level), `strict`
  (aggressive checks only where the user can resolve a block, like memory writes). Important
  rule from the source: *new patterns must anchor on attack vocabulary, not bossy English* —
  "you must" is common in legitimate `AGENTS.md` files.
- **Invisible-Unicode detection** in anything injected into the prompt.
- **Secret redaction** (`agent/redact.py`) in logs and tool output: short tokens fully masked,
  long ones keep first 6 / last 4 characters for debuggability. Connection strings handled
  separately.
- **Secrets vs config separation:** `.env` holds secrets only; behaviour lives in `config.yaml`.
- **URL/website policy** for browser and web tools (`tools/url_safety.py`,
  `tools/website_policy.py`).

**Source:** `tools/threat_patterns.py`, `agent/redact.py`, `agent/redact_connstr.py`,
`tools/url_safety.py`, `SECURITY.md`.

---

## 21. The turn loop: robustness

The loop itself (`agent/conversation_loop.py`, `agent/turn_*.py`) contains several small
patterns that each prevent a class of failures.

- **One internal message format** (OpenAI-style `role/content/tool_calls`) with adapters to
  other APIs at the edge. Reasoning stored separately on the assistant message.
- **Strict role alternation** enforced: never two assistant or two user messages in a row; only
  `tool` messages may repeat.
- **Interruptible API calls:** the HTTP call runs on a worker thread while the main thread waits
  on {response, interrupt event, timeout}. On interrupt (new message, `/stop`), the response is
  discarded — **no partial response is injected into history**.
- **Iteration budget:** a thread-safe consume/refund counter per agent (parent default 500,
  children their own cap). At 100% the agent stops and returns a summary of work done.
- **Error classifier** (`agent/error_classifier.py`): a priority-ordered pipeline maps an API
  exception to a `ClassifiedError` with recovery hints — retry, rotate credential, fall back,
  compress, abort. The retry loop consults hints instead of string-matching errors everywhere.
- **Context-overflow recovery:** a provider "too long" error triggers compression and a retry.
- **Fallback chain:** on 429/5xx/auth errors, try the next configured model/provider; refresh
  credentials before failing over on 401/403.
- **Empty-response guard:** retries an empty completion up to 3 times, but detects
  *deterministic* empties (two in a row with zero output tokens from the same model) and skips
  straight to fallback, since each retry re-bills the full input.
- **Repetition guard:** detects a degenerate repetition loop (long verbatim repeats covering most
  of the output) **before** a `finish_reason=length` "continue" nudge would stitch it into a
  giant response, and aborts with a clear error.
- **Truncation handling:** `finish_reason=length` continues the response in a bounded way.
- **Streaming `<think>` scrubber:** a stateful scrubber that holds partial tags across stream
  deltas so reasoning doesn't leak into the visible answer.
- **Memory flush before anything lossy** (compression, session end).
- **Rich callback surface** for UIs: tool progress, thinking, reasoning, clarify, step, stream
  delta, tool-call-parsed, status. The loop is UI-agnostic; CLI, gateway, TUI and editor
  integrations all plug in via callbacks.

**Source:** `agent/conversation_loop.py`, `agent/turn_*.py`, `agent/error_classifier.py`,
`agent/retry_utils.py`, `agent/empty_response_guard.py`, `agent/repetition_guard.py`,
`agent/think_scrubber.py`, `agent/iteration_budget.py`, `agent/interrupt_control.py`,
`website/docs/developer-guide/agent-loop.md`.

---

## 22. Background processes with completion notifications

**What it is:** the terminal tool can start long-running commands in the background and the
harness tells the model when something happens — instead of the model polling.

**How Hermes does it** (`tools/process_registry*.py`, `tools/terminal_tool_background.py`):
- `terminal(..., background=true)` returns a process id; `process_manage` lists, reads output,
  writes stdin, waits, and kills.
- **`notify_on_complete`**: when the process exits, a notification is injected into the
  conversation (`[IMPORTANT: …]`).
- **`watch_patterns`**: notify when output matches a regex (e.g. "Server listening on").
  Overflow and disable events are reported too.
- Output is logged to files; checkpoints survive restarts.
- The same completion queue delivers **async delegation** results.

**Copy it:** a process registry + a completion queue that the loop drains between turns and
injects as a user-side notification. Removes a whole category of wasteful polling turns.

**Source:** `tools/process_registry.py`, `tools/process_registry_notifications.py`,
`tools/terminal_tool_background.py`.

---

## 23. Planning and interaction tools

**`todo_list`** (`tools/todo_tool.py`):
- One tool: pass `todos` to write, omit to read; **every call returns the full list**.
- State lives on the agent (per session), **re-injected after context compression** so the plan
  survives compaction.
- Every write bumps a monotonic revision so UIs can reject stale updates.
- No system-prompt mutation.

**`clarify`** (`tools/clarify_tool.py`):
- Structured multiple-choice or open-ended questions to the user.
- The UI is a platform-provided callback (CLI prompt, chat buttons, TUI dialog).
- Forces sequential execution in a parallel tool batch; blocked in subagents.

**`/plan`:** writes a Markdown implementation plan to `.hermes/plans/` without executing.

**Source:** `tools/todo_tool.py`, `tools/clarify_tool.py`, `agent/plan_prompt.py`.

---

## 24. Long-running work

Five mechanisms let the agent keep working without the user re-prompting. Each one has a
clear, distinct scope — that separation is itself worth copying.

| Mechanism | Scope | How it works |
|---|---|---|
| **`/goal <text>`** | One session, until done | After every turn a lightweight **judge model** decides `done` / `continue` / `blocked` against the goal. On `continue`, a continuation prompt (with the judge's reason) is fed back into the same session. Turn budget (default 20), `/subgoal` adds criteria mid-run, quality gates, completion contracts, auto-parks while waiting on a background process. The "Ralph loop". |
| **`/loop`** | One session, on an interval | Re-runs a prompt on a cadence, with stop conditions. |
| **`/heartbeat`** | One session, when idle | A prompt that re-enters the session whenever it goes idle ("every 10m check the deployment"). |
| **Cron** | New session per run | Natural-language scheduled jobs with delivery to any channel. Handles missed runs, quota holds, retries, a delivery queue, model pinning, a per-job notepad, blueprints, and the agent can create jobs itself via `cronjob_manage`. Skills referenced by jobs are protected from archival. |
| **Kanban** | Many tasks, many workers | A durable SQLite board: tasks with dependencies, assignees (profiles), claims with heartbeats and stale-claim detection, review / request-changes states, comments, attachments, PR completion contracts. Each card is dispatched to its own worker session. A card can run in goal mode. |

**Why the separation works:** users (and the model) know which tool to reach for, and
features don't accidentally cascade into each other (`/goal` never creates Kanban cards).

**Source:** `website/docs/user-guide/features/{goals,loops,heartbeat,cron,kanban}.md`, `cron/`,
`hermes_cli/kanban*.py`, `tools/kanban_tools.py`, `tools/cronjob_tools.py`.

---

## 25. Steering and conversation control

Small user-facing controls that make a harness pleasant to drive. All are cheap to implement
and cache-friendly.

| Command | Behaviour |
|---|---|
| `/steer <msg>` | Inject a message **after the next tool call** without interrupting the turn. |
| `/queue <msg>` | Queue a prompt for the next turn (list/edit/move/clear queue). |
| `/busy` | Choose what a message sent mid-turn does: queue, steer, or interrupt. |
| `/stop`, or just send a new message | Interrupt the current turn cleanly (§21). |
| `/btw <q>` | Ask a side question **about** the conversation without adding to it: a detached fork replays the snapshot against the warm cache, tools denied, nothing persisted. |
| `/bg <prompt>` | Run a prompt in a separate background session. |
| `/retry`, `/undo [N]` | Resend the last message; back up N user turns. |
| `/rewind` | Pick an earlier prompt to edit & resend, or fork from it. |
| `/branch` | Branch the current session into a new thread. |
| `/compress [here N] [--preview]` | Manual compaction with preview. |
| `/context`, `/usage`, `/insights` | Context-window gauge with category breakdown; token/cost usage; usage analytics. |
| `/diff` | Working-tree git diff (unstaged + untracked shown as additions). |
| `/rollback` | Restore filesystem checkpoints (§18). |
| `/review` | Spawn an independent subagent to review the work just discussed. |
| `/refine` | Run the self-improvement review now (§6). |
| `/journey` | Timeline of everything learned (skills + memory), with edit/delete. |
| `/handoff` | Move the session to another surface (e.g. CLI → Telegram). |

**Source:** `hermes_cli/commands.py` (single declarative command registry with aliases, args
hints and busy policies — itself a good pattern), `agent/side_question.py`, `tools/working_diff.py`.

---

## 26. Lifecycle hooks and plugin surfaces

**Hooks** let users and plugins observe or modify behaviour without touching the core. Hermes
exposes a broad set; the most useful to copy:

- **Turn:** `on_turn_start`, `on_turn_complete`, `pre_llm_call` (return extra context → appended
  to the user message, never the system prompt), `post_llm_call`, `post_api_request`.
- **Tools:** `pre_tool_call` (can block/modify), `post_tool_call`.
- **Session:** `on_session_start`, `on_session_end`, `on_session_reset`, `on_session_switch`,
  `on_session_finalize`.
- **Context:** `on_pre_compress`, `on_compaction`, `post_compaction`.
- **Learning:** `on_memory_write`, `on_skill_lifecycle`.
- **Streaming:** `on_stream_start`, `on_text_delta`, `on_reasoning_delta`, `on_stream_end`.
- **Delegation / human input:** `on_delegation`, `on_human_input_request`,
  `on_human_input_resolved`.

Design rules worth copying:
- Oversized hook output is **spilled to a file** with a head/tail preview (hooks can't flood
  context).
- Shell hooks (run a command on an event) for users who don't write Python.
- Outbound webhooks are opt-in.
- **"No hooks without a concrete consumer"** — a contribution rule that keeps the hook surface
  from growing speculatively.

**Plugin kinds** are ABCs with registration: memory providers, context engines, web search,
image/video generation, browser backends, terminal environments, delegation backends, secret
sources, config backends, platforms. Each is a narrow interface the core orchestrates.

**Source:** `hermes_cli/plugins*.py`, `agent/shell_hooks.py`, `agent/outbound_webhooks.py`,
`tools/hook_output_spill.py`, `website/docs/user-guide/features/hooks.md`,
`website/docs/developer-guide/observer-hooks.md`, `website/docs/developer-guide/middleware.md`.

---

## 27. Auxiliary models for side tasks

**What it is:** a separate, configurable model slot per side task, so expensive main models
aren't used for cheap work.

**How Hermes does it** (`agent/auxiliary_client.py`): task slots such as `compression`,
`vision`, `session_search`, `curator`, `side_question`, title generation, background review and
smart approval. Each slot defaults to "auto" (= the main model) and can be pinned to a cheaper
provider/model, with **its own fallback chain**. Usage is accounted back to the session.

**Rule of thumb from the docs:** the compression model must have a context window at least as
large as the main model's, because the whole middle is sent in one call.

**Source:** `agent/auxiliary_client.py`, `agent/auxiliary_task_config.py`,
`agent/auxiliary_*.py`.

---

## 28. Profiles: isolated agent identities

**What it is:** several fully isolated agent "homes" (config, memory, skills, sessions,
secrets) on one machine — e.g. `work` and `personal`.

**How Hermes does it:**
- Every path goes through one resolver, `get_hermes_home()`, which is profile-aware; nothing
  hardcodes `~/.hermes` (enforced by a lint rule).
- Profiles are **islands on purpose**: memory written in one profile isn't visible in another.
- One process can serve many profiles; any code running outside a turn (timers, callbacks,
  thread hops, child processes) must bind the owning profile explicitly. An unbound read of a
  global is treated as a bug (a "silent default-profile leak").
- Export/import a profile as an archive.
- "One agent per home": two processes writing the same memory files would compound each
  other's entries.

**Copy it:** route every state path through one function from day one. Retrofitting this is
painful.

**Source:** `hermes_constants.py`, `website/docs/user-guide/profiles.md`, `gateway/AGENTS.md`
§ Profile scope.

---

## 29. Prompt-surface snapshot tests

**What it is:** a CI check that renders the **exact** system prompt and tool schemas a stock
install sends, and diffs them against a committed snapshot.

**How Hermes does it:** `scripts/ci/prompt_surface.py render` builds the prompt through the
production assembly code for several surfaces/model families and writes fixtures to
`tests/fixtures/prompt_surface/`. A test fails if the snapshot is stale; a CI job posts
per-surface token deltas and added/removed tools/parameters on the PR; changes to the fixtures
need core-maintainer approval.

**Why it works:** a one-line code change can silently rewrite the prompt or add a tool. Making
the prompt surface a reviewed artifact catches cost and behaviour regressions before merge.

**Source:** `scripts/ci/prompt_surface.py`, `tests/ci/test_prompt_surface.py`.

---

## 30. Suggested adoption order

If you're building a harness from scratch, this order gives the most value earliest:

1. **Cache-stable prompt assembly** (§2) + append-only history (§10). Everything else assumes it.
2. **Tool registry with `check_fn` + toolsets** (§12), uniform JSON errors, parallel calls.
3. **File-editing safety** (§16) + **output truncation/spill** (§17) + **checkpoints** (§18).
4. **Dangerous-command approval** (§19).
5. **Bounded memory** (§3) + **session store with FTS search** (§4).
6. **Context compression** — at least phase 1 pruning + structured summary (§9).
7. **Skills with progressive disclosure** + `skill_manage` (§5).
8. **Background self-improvement review** (§6) — copy the prompts.
9. **Todo + clarify** (§23), **interrupts + budget + error classifier + guards** (§21).
10. **Subagents** (§15) and **`execute_code`** (§14).
11. **Background process notifications** (§22), **`/goal`** (§24), **steering** (§25).
12. **Curator** (§7), **tool search** (§13), **hooks/plugins** (§26), **prompt-surface tests** (§29).

### The ten most transferable ideas

1. Freeze the system prompt per session; deliver mid-session changes as one-shot notes in the
   user message.
2. Hard-capped memory whose overflow error tells the model exactly how much to free.
3. Memory = who the user is and facts about the environment; skills = how to do a class of task.
   One fact, one place.
4. Progressive disclosure: one-line index → full skill → support file.
5. A post-turn, cache-warm background fork that reviews the turn and updates memory/skills, with
   explicit "do not capture" rules (especially: never save negative claims about tools).
6. Compression: prune old tool outputs first, keep tool pairs intact, a structured summary,
   update it on re-compression.
7. Context files found mid-session are attached to the **tool result**, not the system prompt.
8. Refuse whole-file overwrites of content the agent never read or that changed since.
9. Only send tools whose `check_fn` passes, and rewrite cross-referencing schemas to match.
10. Inject completion notifications for background work instead of letting the model poll.
