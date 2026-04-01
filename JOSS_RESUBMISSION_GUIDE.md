# MicroPyzzotMet: JOSS Resubmission Guide

This file is the single project-management note for the JOSS resubmission. The user-facing documentation should live in `README.md` and `docs/`, not in multiple overlapping Markdown guides at the repository root.

## Current Status

### Paper

- `JOSS/paper.md` contains the submission manuscript.
- The paper includes the required state-of-the-field, software design, and research-impact framing.
- AI disclosure text has been updated and should be checked one more time before final resubmission for wording consistency.

### Repository Quality Signals

- `README.md` is the primary entry point for installation and usage.
- `CONTRIBUTING.md` documents contribution expectations.
- `tests/` contains automated tests.
- `.github/workflows/tests.yml` runs CI.
- `.github/workflows/publish.yml` is available for release publishing.
- `.readthedocs.yml` and `docs/` are configured, and Read the Docs now builds successfully.

### Packaging

- The package uses a `src/` layout.
- The CLI entry point is defined in `pyproject.toml`.
- The project is installable with `pip install -e .`.

## What This Guide Covers

This guide should be used for three things only:

1. Track which JOSS concerns have been addressed.
2. Record the remaining pre-resubmission actions.
3. Provide a short checklist for the final submission pass.

## Addressed JOSS Concerns

### 1. Documentation and discoverability

- Main documentation exists in `README.md`.
- Hosted documentation is configured through Read the Docs.
- Sphinx API pages point to package-qualified modules and build successfully.

### 2. Tests and CI

- The repository contains tests for imports, CLI behavior, and selected utilities.
- GitHub Actions CI is configured and should be kept green before resubmission.

### 3. Open-source development practices

- Contribution guidance is present in `CONTRIBUTING.md`.
- The repository has a conventional package structure and automated checks.

### 4. Release readiness

- Packaging metadata is in place.
- A release workflow exists.
- The remaining step is to make sure the tagged release used in the JOSS resubmission is the intended one.

## Remaining Actions Before Resubmission

### Required

1. Run the test suite and confirm CI is green.
2. Confirm the version number used in the package and manuscript is the intended release version.
3. Create or verify the release tag that you want to cite in the JOSS review.
4. Re-read `JOSS/paper.md` and make sure the wording matches the current repository state.

### Recommended

1. Verify the Read the Docs site from the public URL after the latest push.
2. Verify the GitHub release page includes short release notes.
3. If PyPI publication is desired for the submission narrative, publish the same tagged release.

## Final Resubmission Checklist

- [ ] `JOSS/paper.md` reflects the current software state.
- [ ] `README.md` is the canonical setup and usage guide.
- [ ] `CONTRIBUTING.md` is still accurate.
- [ ] Tests pass locally and in CI.
- [ ] Read the Docs build succeeds.
- [ ] Release tag exists and is the one you want to cite.
- [ ] Submission comments briefly summarize what changed since the previous round.

## Suggested Resubmission Summary

Use a short summary in the JOSS discussion along these lines:

> We consolidated the repository documentation, retained `README.md` as the primary user guide, kept `CONTRIBUTING.md` for development practices, and maintained a single `JOSS_RESUBMISSION_GUIDE.md` for submission tracking. The repository includes automated tests, CI, packaged installation, and a working Read the Docs build.

## Canonical Files

These should be treated as the authoritative files for the resubmission:

- `README.md`
- `CONTRIBUTING.md`
- `JOSS_RESUBMISSION_GUIDE.md`
- `JOSS/paper.md`
- `.readthedocs.yml`
- `docs/`
- `pyproject.toml`
