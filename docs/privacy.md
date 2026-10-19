# Privacy and threat model

This is a small research library, not a compliance product. This page states
what it does and does not protect.

## What is stored

- `nodes`: summary, scene, tags, frozen observation scores, source quote,
  source turn, and observation source.
- `edges`: link weights and optional evidence quotes.
- `self_trace`: retrieval and proposal fingerprints, node ids, and observation
  scores. New retrieval/proposal rows do not store the original query, scene,
  summary, source quote, or note.
- `cognition_edit` rows may contain a caller-supplied reason/detail. Treat them
  as free text and delete them explicitly if they contain personal data.

The repository itself ships only synthetic examples. Runtime data is whatever
the caller writes.

## Deletion

`MemoryStore.forget()` removes nodes, incident edges, tags, source quotes, and
the trace rows that reference those nodes or contain their deleted text. It
leaves a tombstone with an id, kind, timestamp, and reason, but no memory
content. `scrub()` runs `VACUUM` on a file database.

External caches and logs are outside the store. Use
`find_text_in_files()` to locate them and `scrub_files()` to remove matching
lines. `scrub_files()` does not keep a `.bak` by default; asking for
`backup=True` deliberately keeps the original bytes and must be followed by
your own secure-deletion step.

Deletion cannot reach OS snapshots, filesystem journals, SSD wear-leveling
copies, or backups you made elsewhere. Full-disk encryption and a retention
policy are still required.

## At-rest protection

- The core uses plaintext SQLite and has no built-in field or database
  encryption.
- On POSIX, the library asks for `0600` on the database, exports, and scrub
  targets. On Windows, `chmod` is best-effort; configure NTFS ACLs yourself.
- For at-rest encryption, put the database on a BitLocker/LUKS volume or use a
  SQLCipher-compatible SQLite build. Backups are plaintext unless you encrypt
  the destination.
- Keep the database in a private directory and do not ship it with logs,
  caches, or model checkpoints.

## Not provided

- consent management, age gating, or jurisdiction-specific policy;
- general PII detection beyond credential/ID/card patterns;
- encryption or key management;
- protection against a compromised host, debugger, or administrator account.

If you process other people's conversations, obtain a lawful basis, give
notice, minimize collection, set a retention period, and provide a human
review and deletion path.
