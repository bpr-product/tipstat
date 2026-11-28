# Memory Hackathon: participant spec

## The problem

You're building the **memory and agent of a personal assistant**. A person talks to their
assistant: work, family, plans that change. Around those chats things happen in their apps:
Slack messages, emails, calendar changes, ticket updates. Your system receives all of it
**one message at a time, in the order it happened**, like a live assistant.

At points along the way we ask a question ("anything important coming up?") or give a job
("set up a 1:1 with Nikhil on Friday"). Your system has to:
- remember what matters
- keep up with changes, including facts that only ever arrived through an app
- know what it doesn't know, and forget what it was asked to forget
- **act correctly through tools** using all of that

You get **persona A** to develop against: 43 messages, 20 questions and 8 jobs. You're scored
on a **hidden persona B**: a different person, **about 175 messages (about 37,000 tokens)**,
many of them long, with about 46 questions and 23 jobs.

## What you build

An HTTP service with five endpoints (start from the starter template):

| Endpoint | Body | Returns |
|---|---|---|
| `GET /health` | | `{"ok": true}` |
| `POST /reset` | `{}` | `{"ok": true}`: wipe all memory |
| `POST /message` | `{"message": Message}` | `{"ok": true}` |
| `POST /answer` | `{"question"}` | `{"answer", "evidence", "trace"}` |
| `POST /act` | `{"request", "mcp_url", "tools_url"}` | `{"reply", "trace"}` |

- **Messages arrive one at a time, in time order.** We send the next one only after your call
  returns. Questions and jobs come between messages, so you can never see the future.
- **"Now" is the timestamp of the latest message you received.** Use it for "today",
  "tomorrow" and "this week". Nothing else tells you the time.
- A question may be about the message you received a moment ago, or about one from weeks
  ago. Finish updating memory before `/message` returns.
- `/reset` always comes first. Handle a question even when you have no memories yet.
- If a `/message` call fails or times out, we move on to the next one; it is not sent again.
- We send one request at a time.
- **`evidence`** is `[{"message_id": "m012", "text": "..."}]`: what you actually used.

**Message: a chat** between the user and their assistant (a short exchange)
```json
{"id": "m012", "ts": "2026-11-03T19:50:00+05:30", "type": "chat",
 "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```
**Message: an app event** (one thing that happened in Slack, calendar, email, docs or tickets)
```json
{"id": "m013", "ts": "2026-11-03T20:25:00+05:30", "type": "event",
 "source": "inbound", "op": "add_message",
 "data": {"channel": "#payments-team", "from": "@nikhil", "text": "..."}}
```
`source` is `inbound` (other people), `user` (the user acting in an app themself: a DM or
email they sent, an RSVP, a ticket they moved) or `assistant`.

| op | data |
|---|---|
| `add_message` | `id, from, text, ts` + `channel` or `to`; `thread_id` for a thread reply |
| `add_person` | a person record |
| `add_event`, `update_event`, `cancel_event` | event `id, title, start, end, ...` / changed fields / `id` |
| `rsvp_event` | `id, response` (accepted, tentative, declined), `note?` |
| `add_email` | `id, thread_id, folder, from, to, cc, subject, body, ts, labels` |
| `add_doc`, `update_doc`, `add_doc_comment` | a doc / `id` + `append` or `body` / `doc_id, author, text, ts` |
| `add_task`, `update_task`, `add_task_comment` | a ticket / `key` + changed fields + `comment?` |
| `add_reminder`, `complete_reminder` | a reminder / `id` |

**Some facts only ever arrive as app events**, including things the user said *to other
people*. The mock world also has a **backlog** (older messages, emails, docs, tickets and the
calendar) that is never sent to `/message`; it is only visible through the tools during `/act`.

## The mock world (tools)

| App | Tools |
|---|---|
| Slack | `slack_list_channels`, `slack_read_messages`, `slack_read_thread`, `slack_search_messages`, `slack_send_message`, `slack_schedule_message` |
| Calendar | `calendar_list_events`, `calendar_get_event`, `calendar_find_free_slots`, `calendar_create_event`, `calendar_update_event`, `calendar_cancel_event`, `calendar_respond` |
| Email | `email_list`, `email_search`, `email_read_thread`, `email_send`, `email_reply` |
| Docs | `docs_search`, `docs_read`, `docs_comment` |
| Tasks | `tasks_list`, `tasks_get`, `tasks_search`, `tasks_update` |
| Reminders | `reminders_create`, `reminders_list`, `reminders_complete` |
| Contacts | `contacts_lookup` |

- Every list/search returns **at most 50** items. `slack_read_messages` and `email_list` page back
  with `before=<ts or the oldest id you got>`. List views
  clip long text (event descriptions, email bodies); `calendar_get_event`,
  `email_read_thread` and `docs_read` (6,000 characters per call, `next_offset`) give it in full.
- `calendar_cancel_event` cancels a meeting **for everyone**; to drop out of someone
  else's meeting, `calendar_respond` with `declined`.
- All 29 schemas are ~2,400 tokens and count against your 8,000 per-call cap. `GET {tools_url}?apps=slack,calendar`
  (or MCP `tools/list` with `{"apps": "slack,calendar"}`) returns a subset.
- The tools work **only during `/act`**; at other times they return an error. During a job
  the world shows what existed at that moment, with its clock at the latest message's time.

- **MCP** (Streamable HTTP): `POST {mcp_url}`. Works with the official MCP SDKs.
- **Plain HTTP:** `GET {tools_url}` returns OpenAI function-calling schemas, and
  `POST {tools_url}/<tool_name>` with a JSON body calls a tool.
- `calendar_find_free_slots` only knows the **calendar**. It doesn't know about the
  flight the user mentioned in chat. Your memory has to.
- Run it locally: `python3 -m world.server --port 8800 --persona data/dev/persona_a --date 2026-11-03`,
  then `python3 hello_world.py http://localhost:8800`.

## Limits

**One model**, `deepseek/deepseek-v4.1-flash`, through `OPENAI_BASE_URL` + `OPENAI_API_KEY`.

**Every model call: at most 8,000 input tokens** (413 above it). Each kind of request also has
a budget, shared by all its model calls (429 above it):

| Request | Model calls | Input tokens in total | Time |
|---|---|---|---|
| each `/message` | up to 2 | 10,000 | 30 s (all messages together: 20 min) |
| each `/answer` | up to 2 | **4,000** | 30 s |
| each `/act` | up to 12 | 60,000 | 90 s |

- **You choose how to remember.** You may call the model on every message, on some, or on
  none. You don't have to use two calls.
- The hidden history is about **9 times** a question's budget, so you can't send it all with
  a question.
- Whole run: 30 minutes, 4 CPUs, 6 GB RAM, no GPU. If time runs out, what finished is scored.

## Rules

1. **Counting.** The 8,000 and the budgets count everything the model reads: messages, tool
   calls, tool results and tool schemas, with non-ASCII characters at one token each. The
   gateway also refuses other models, `models`, `plugins`, `provider`, `transforms`,
   `prompt`, non-text content and `n` > 1.
2. **Coding tools: any tool, allowed models only, your design.** Code only on the allowed
   coding models (RULES.md section 4); the memory design must come from your own prompts.
   Export your sessions into `prompt_history/`.
3. **Any language, any framework.** The harness only sees your HTTP API. Use the Chat
   Completions API through `OPENAI_BASE_URL`; tested setups are in the kit README.
4. **No agent-memory products** (Mem0, Supermemory, Letta/MemGPT, Zep/Graphiti, Cognee,
   LangMem, framework memory classes). Databases, vector DBs, embeddings, rerankers, BM25 and
   agent frameworks for the tool loop are fine.
5. **Packaging.** `docker compose up --build` must start everything (if you add `compose.extra.yml`,
   we run `docker compose -f docker-compose.yml -f compose.extra.yml up --build`), with service
   `memory` on `${PORT:-8000}:8000`. It runs **offline** (only the model gateway and mock world are
   reachable): download models in an optional `build.sh`, which runs once at build time.
6. **Submit by 17:00:** your repo with `docker-compose.yml`, `Dockerfile`, `MEMORY.md` and
   `prompt_history/`. Run `python3 scripts/check_submission.py .` first.

## Scoring

**Memory test score = 60% questions + 40% jobs**, each a difficulty-weighted average, x100. The
memory test is 30% of your event score (RULES.md section 11).

**Questions.** An LLM judge grades the meaning of your answer, not the exact wording.

| Part | What | Weight |
|---|---|---|
| Correctness | CORRECT 1 / PARTIAL 0.5 / WRONG 0 | 70% |
| Grounding 0-2 | Does your `evidence`/`trace` actually support the answer? | 15% |
| Usefulness 0-2 | Thoughtful assistant behaviour (flags importance, pending to-dos, what changed) | 15% |

A **HALLUCINATED** answer scores -0.25. That covers confidently wrong answers, old values
given as current, cancelled plans presented as still on, someone else's fact attributed to
the user, or revealing something you were told to forget.

**Actions.** We diff the world before and after your action.

| Part | What | Weight |
|---|---|---|
| Hard checks | Deterministic: right recipient, event on the right day/time, no clash with a known commitment, updated rather than duplicated, reminder created... | 60% |
| Judge | Outcome label (70%) plus process quality, i.e. sensible tool use and a clear reply (30%) | 40% |

A **HARMFUL** action scores -0.25. That covers a wrong recipient, leaking forgotten or
private info, acting on a cancelled plan, double-booking over something you should
remember, or ignoring a standing instruction like "never message Nikhil after 9 pm".
Doing nothing never scores well.

Prizes follow RULES.md section 11.

## What we test

| Questions | Actions |
|---|---|
| Facts the user stated to *other people* (DMs they sent), or that live only in an email, doc edit, ticket or RSVP | Replying in the right email thread; declining rather than cancelling; updating the right ticket |
| Importance recall, pending to-dos, low-salience mentions | Resolving the right person among similar names |
| Knowledge updates, reschedules, cancellations | Scheduling around commitments the calendar doesn't know |
| Forget requests, attribution (whose fact is it) | Acting on the latest value (gym moved to 7 pm) |
| Facts that arrived only via Slack/Calendar | Not acting on cancelled plans |
| Temporal reasoning, routine exceptions, counting | Not leaking forgotten info in messages |
| Multi-hop, hypothetical vs actual, abstention | Follow-through (reminders), updating instead of duplicating, standing preferences |

## Local loop

```bash
docker compose up --build        # or: python3 server.py
python3 -m harness.run --system http://localhost:8000 --data data/dev/persona_a \
    --world auto --judge offline --out runs/try1
python3 scripts/check_submission.py . --live http://localhost:8000
```

`--world auto` starts a private mock world for the run (not reachable from Docker: with
Docker, run `python3 -m world.server --port 8800` and pass `--world http://localhost:8800/t/me
--world-public http://host.docker.internal:8800/t/me`). `--judge offline` is a crude keyword
check that needs no key. Limits are enforced only through the gateway: add `--proxy` as the
kit README shows. On Windows, use `python` where macOS needs `python3`.
