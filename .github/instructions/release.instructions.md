---
applyTo: "VERSION,CHANGELOG.md,.github/workflows/**,docs/pull_request_template.md"
---

# Orchestrator release and repository flow

Read `.github/copilot-instructions.md` and load `orchestrator-release` before
changing a release artifact.

## Normal branches and pull requests

- Development starts from `develop`.
- Implementation branches use `feature/<short-description>` and target
  `develop`.
- Release branches start from `develop`, use `release/x.y.z`, and target
  `main`.
- A feature branch targets `main` only when a maintainer explicitly authorizes
  that exception for the current task.
- Pull request descriptions state purpose, work type, target branch,
  validation, documentation impact, compatibility, and required follow-up.

## Version identity

Use semantic versioning: `MAJOR.MINOR.PATCH`.

- Release branch: `release/X.Y.Z`
- Root `VERSION`: `X.Y.Z`
- Changelog heading: `## [vX.Y.Z] - YYYY-MM-DD`
- Git tag: `vX.Y.Z`
- GitHub Release title: exactly `vX.Y.Z`

Never add `v` to `VERSION` or the release branch. Never omit `v` from the tag,
release title, or versioned changelog heading.

- PATCH: bug fixes and minor compatible improvements.
- MINOR: backward-compatible features.
- MAJOR: breaking changes.

## VERSION lifecycle

- On `develop`, `VERSION` represents the latest version already present in
  that branch.
- Feature work does not preemptively change `VERSION`.
- On `release/X.Y.Z`, update `VERSION` to exactly `X.Y.Z`.
- Do not leave the release branch name, `VERSION`, changelog, tag, or release
  title out of sync.

## Changelog lifecycle

- `develop` contains exactly one top-level `## [Unreleased]` section.
- Record every new change on `develop` under that section without assigning a
  future version.
- A release branch replaces `[Unreleased]` with
  `## [vX.Y.Z] - YYYY-MM-DD`.
- A release branch and `main` must never contain `[Unreleased]`.
- After the release branch is pushed, return to `develop`, add one new empty
  `## [Unreleased]` section at the top, commit it, and push `develop`.

Use applicable Keep a Changelog sections:

- `Added`
- `Changed`
- `Fixed`
- `Removed`

Each entry starts with a bold technical title and explains what changed and
why it matters. Avoid vague entries such as "minor updates", "improvements",
or "fixes".

## Release safety

- Do not mix feature work into a release branch.
- Do not infer a version from stale prose or a previous release.
- Keep release notes, changelog, and operator documentation consistent.
- Never expose credentials, tokens, personal data, private validation
  environment names, or resource group names in release artifacts.
- Preserve Markdown headings, lists, and tables when editing release notes
  through an API; re-fetch and verify the published body.
- Do not create or modify a tag, GitHub Release, package, image, or production
  deployment without explicit human approval.

Use clear commits such as:

- `feat: add conversation metadata support`
- `fix: correct chat history persistence`
- `docs: update changelog for v3.9.0`
- `chore: prepare release 3.9.0`
