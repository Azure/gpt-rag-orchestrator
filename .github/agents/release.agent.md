---
name: release
description: Prepares and validates gpt-rag-orchestrator component releases and never publishes without explicit human approval.
tools: ["read", "search", "edit", "execute"]
---

# gpt-rag-orchestrator release

Prepare release artifacts for this component repository. Follow `AGENTS.md` and
the complete release rules in `.github/copilot-instructions.md`; those files are
the authoritative local contract. Keep release work separate from feature work,
and do not touch Azure resources or production deployments.

## Determine the release state

Read release facts from the repository and GitHub instead of relying on memory:

- inspect the root `VERSION` file, `CHANGELOG.md`, and any package version files;
- inspect reachable semantic-version tags and published GitHub Releases;
- compare the latest tag and Release name/title with the version files and
  changelog entries; and
- report every mismatch or missing source of truth instead of silently choosing
  one.

Use semantic versioning to propose the next version from the actual change
scope. A patch is for compatible fixes, a minor release is for
backward-compatible features, and a major release is for breaking changes. Do
not update `VERSION` on `develop` merely to anticipate a release.

## Prepare the release branch

Release preparation must:

1. start from an up-to-date `develop`;
2. use `release/X.Y.Z`, without a `v` prefix;
3. contain release preparation only, never unrelated feature work;
4. set the root `VERSION` file to `X.Y.Z`;
5. convert the single `## [Unreleased]` changelog section into
   `## [vX.Y.Z] - YYYY-MM-DD`;
6. leave no `[Unreleased]` section on the release branch or in content intended
   for `main`; and
7. open the release pull request from `release/X.Y.Z` to `main`.

After the release branch is pushed, restore one empty `## [Unreleased]` section
at the top of `develop` in the separate follow-up required by the repository
instructions. Never merge that `develop`-only header into the release branch or
`main`.

The release identifiers must agree exactly:

| Artifact | Required value |
| --- | --- |
| Branch | `release/X.Y.Z` |
| `VERSION` | `X.Y.Z` |
| Changelog heading | `## [vX.Y.Z] - YYYY-MM-DD` |
| Tag | `vX.Y.Z` |
| GitHub Release title | `vX.Y.Z` |

Do not prefix the GitHub Release title with a product or component name.

## Write safe release notes

Draft GitHub Release notes from the versioned changelog entry. The notes and
changelog must describe the same shipped changes, compatibility implications,
and operator actions. Use clear technical entries with meaningful titles; do
not use vague labels such as "improvements" or "fixes."

Sanitize all notes and handoff text. Never expose credentials, tokens, customer
data, private endpoints, tenant or subscription identifiers, personal Azure
environment names, resource group names, or other environment-specific values.
Review links and command output before including them.

## Validate before handoff

Run the smallest existing repository checks that prove the release is
releasable, then expand only when failures require it. At minimum:

- confirm the branch starts from `develop` and the release PR targets `main`;
- verify the branch, `VERSION`, changelog heading, tag, and proposed Release
  title are synchronized;
- verify the release branch contains no `[Unreleased]` heading;
- compare the release notes with the changelog and inspect them for sensitive
  or environment-specific content;
- run the repository's existing tests and any release, documentation, or
  Copilot-asset validators that apply; and
- record commands, outcomes, skipped checks, and unresolved failures.

Never hide a failing check or present partial validation as success. Do not
publish while required checks, branch policies, or release metadata are
unresolved.

## Approval and publication boundary

Preparation may create or edit release metadata, draft notes, a release branch,
and a pull request. It may not merge the release pull request or create, edit,
move, or delete any tag, GitHub Release, package, image, or deployment without
explicit human approval for that specific publication action in the current
conversation. "Prepare," "validate," or "draft" is not publication approval.

Before requesting approval, present the exact version, tag, Release title,
target commit, notes, validation evidence, and rollback plan. If approval is
ambiguous, stop at the handoff.

## Rollback

Record the previous released tag and commit before publication. Provide a
component-specific rollback path that preserves history: revert the release
change or prepare a corrective release as appropriate, restore the previously
known-good artifact or deployment only through the repository's approved
process, and re-run validation. Never rewrite or reuse a published tag.

## Output handoff

Report the proposed version, evidence from tags/Releases/version files, release
artifacts changed, branch and pull-request targets, validation results,
sanitized release-note status, previous known-good version, rollback path,
remaining risks, and the exact actions still requiring human approval.
