# Security policy

This is a `0.x` research preview. Security fixes are applied to the latest
`main` branch; older commits are not supported.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting:

1. Open the repository's **Security** tab.
2. Choose **Report a vulnerability**.

Do not open a public issue containing API keys, credentials, personal data,
runtime databases, or a working exploit against a deployed system. If private
reporting is unavailable, open a minimal public issue asking for a private
channel; do not include the sensitive details.

## In scope

- bypasses of the sensitive-content gate in node, edge, or trace writes;
- deletion/erasure failures, including trace rows or evidence quotes that
  survive `MemoryStore.forget()`;
- raw query/proposal/scene text that appears in `self_trace` despite the
  fingerprint protocol;
- file-encoding corruption or unintended `.bak` retention in `scrub_files()`;
- SQLite path/permission handling that exposes data to other local users;
- dependency or GitHub Actions supply-chain issues.

## Out of scope

- host compromise, debugger access, administrator access, or stolen backups;
- OS snapshots, filesystem journals, SSD wear-leveling copies, and
  third-party caches;
- jurisdiction-specific compliance or consent management;
- model quality or emotional-accuracy claims.

See [docs/privacy.md](docs/privacy.md) for the threat model and at-rest
limitations. Backups are plaintext unless the destination itself is encrypted.
