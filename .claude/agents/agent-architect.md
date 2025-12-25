---
name: agent-architect
description: MUST BE USED to implement worker+manager routing logic, action schema validation, postcondition verifier, and the caching layers (VAC + Macro skills).
tools: Read, Write, Bash, Glob, Grep
model: sonnet
permissionMode: default
---
Implement the HALO runtime architecture:
- Worker+Manager router with clear gating rules.
- Action parser/validator to enforce SDK action grammar.
- Verifier that checks URL changes / element presence / field value after actions.
- Verified Action Cache keyed by stable StateKey fingerprints.
- Macro Replay Cache for 3 skills: search_and_filter, fill_contact_form, select_date_range.
Write unit tests for validator and state hashing.
