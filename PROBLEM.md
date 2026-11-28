# The problem

## A personal assistant that remembers

Imagine an assistant someone talks to all day, every day: about work, family, plans, the gym,
what they're watching, what's worrying them. Around those chats their working life keeps moving
on its own - colleagues post in Slack channels and threads, DMs arrive and get sent, the inbox
fills with notifications, newsletters and receipts, meetings get scheduled, moved and declined,
documents get edited, tickets change hands. Much of it is noise. Some of it matters, sometimes
weeks later.

**You are building that assistant's long-term memory - and the agent that acts on it.**

You can't keep re-reading everything. Your system receives the history **one message at a time**,
in the order it happened, and has to:

| | What it means |
|---|---|
| **Remember** | Keep what matters and skip the rest. Know who said what - the user, someone else, or text the user pasted. Keep a source for everything you hold. |
| **Update** | Notice when something stops being true, and let the new version win - including everything that was derived from the old one. |
| **Recall & forget** | Answer when it helps, say "I don't know" when it was never said, and when asked to forget something, forget it completely. |
| **Act** | Carry out requests through tools (send a message, book or decline a meeting, reply to an email, update a ticket, set a reminder) correctly, using what it remembers. |

## What you get

**A practice persona** (`data/dev/persona_a/`): 43 messages from one person's life - short chats
with the assistant and things that happened in their apps - plus sample questions and sample jobs
with the checks we run on them. Read it first: `persona.json`, then `stream.jsonl` (the messages in
order), then `questions.jsonl` and `actions.jsonl`. Each says which message it comes `after`.

**A mock world** (`world/`): the person's Slack, calendar, email, docs, ticket tracker, reminders
and contacts, served two ways - **MCP** (Streamable HTTP, works with the official MCP SDKs) and
**plain HTTP** (OpenAI function-calling schemas + one POST per tool). Run it on your laptop:

```bash
python3 -m world.server --port 8800 --persona data/dev/persona_a --date 2026-11-03
python3 hello_world.py http://localhost:8800
```

Your system never talks to the real apps. Things that happen in them reach you in two ways: as
**app event messages** in the stream (a message arrived, an invite was declined, a ticket moved),
and through the **tools** while you carry out a job. Some facts only ever arrive as app events,
never in chat.

**The evaluation harness, judge and LLM gateway** (`harness/`) - exactly what we score you with.

## What we run you against

A **hidden persona**: a different person, about **175 messages** (about 37,000 tokens), many of
them long. It has the same shape as the practice persona, only longer and harder. We send it to
your system one message at a time, and between messages we:

- **ask questions** - "now" is the time of the latest message; some are about something you just
  received, others about something from weeks earlier; and
- **give jobs** in the mock world, as it is at that moment, then check what your system changed.

Answers are graded by an LLM judge on meaning, on whether your returned evidence supports them,
and on usefulness. Actions are graded on the state of the world afterwards plus a judge. A
confident wrong answer, or a harmful action, scores below zero; "I don't know" scores zero.

## The constraints that make it interesting

- **One model at runtime** for everyone, through our gateway, **8,000 input tokens per call**.
  A question may use **4,000 input tokens in total**, so you can't send the whole history with it.
  Memory design is the whole game.
- **How you remember is your choice**: a model call on every message (up to 2), on some, or none.
- **Time and machine limits**: 20 minutes for all the messages, 30 seconds per question, 4 CPUs /
  6 GB, offline apart from the gateway and the mock world.
- **Your design, your code**: no off-the-shelf agent-memory products; any database, retrieval
  method or local model you build with is fine.

Everything precise - endpoints, payloads, tools, budgets, scoring - is in [SPEC.md](SPEC.md) and
[RULES.md](RULES.md).

## What you hand in

A public repository (created from this template) that starts with `docker compose up --build`
and serves the API on port 8000, a one-page `MEMORY.md` explaining your design, and your coding
sessions in `prompt_history/`. Finalists demo live and walk the judges through their design.
