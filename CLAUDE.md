# CLAUDE.md

Instructions for LLM agents working in this repository. Read
[CONTRIBUTING.md](CONTRIBUTING.md) first; its rules also apply to agents.

@CONTRIBUTING.md

These rules override default model behaviour.

## Scope

`archimono` solves fixed-composition MAX-CUT on periodic graphs from the 11
Archimedean tilings. It is a combinatorial and geometric library.

Describe behaviour in terms of graphs, cuts, compositions, supercells, and
counts. Do not import a physical model: energy, entropy, spin, and ground
states are the caller's interpretation, and do not belong in code, docstrings,
comments, commit messages, or issue and PR text. Justify design decisions with
combinatorial facts, which are checkable against the code.

## Terminology

Use the vocabulary already in the codebase; synonyms fragment search.

- **edge**, never "bond" (issue #81, PR #82)
- **supercell**, **sublattice**, **HNF matrix**
- **optima**, `n_optimal_assignments`
- **vertex**: a point of a tiling's primitive cell
- **node**: a point of a supercell graph ("vertex ordering" remains the term
  for a node processing order)
- **site**: an index acted on by a symmetry permutation
- **atom**: an entry of an ASE `Atoms` object, in export and visualisation only

Describe a property as belonging to the object that has it: a tiling or
graph is bipartite, an assignment has a cut value, and a solver only produces
that assignment.

## Environment

The dev env is the conda env `archimono`, provisioned with
`pip install -e ".[dev,docs]"`. Do not probe for installed packages or suggest
`pip install`.

Bare `python`, `python3`, or `pip` may resolve to a different interpreter that
lacks the dependencies. Run tooling through the env:

```bash
conda run -n archimono python -m ruff check src/ scripts/ tests/
conda run -n archimono python -m mypy src/
conda run -n archimono python -m pytest              # fast + contract
conda run -n archimono python -m pytest -m fast
conda run -n archimono python -m pytest -m contract
conda run -n archimono python -m pytest -m heavy
conda run -n archimono python -m pytest -m ""        # all tiers
conda run -n archimono python -m pytest -k "test_name"
```

## Verification

Run the command and read its output before calling anything done, fixed, or
passing. Report failures and skipped steps.

Before a commit or PR: `ruff check src/ scripts/ tests/`, `mypy src/`, then
`pytest -m fast`, `-m contract`, `-m heavy`. Say which tiers ran.

Recompute changed expected values by running the code, not by adjusting a
number until the test passes.

## Written output

Covers chat, review, commit messages, and issue and PR text.

- State facts plainly. No dramatic phrasing or emphasis padding.
- Lead with the result.
- A claim about code behaviour carries a runnable snippet and its verbatim
  output.
- Correct your own errors in one sentence, then continue.
- Review findings address the change, not the author.

## Git and GitHub

- Branch per issue or fix; do not commit to `main` directly.
- Issue titles short and specific, always labelled.
- A PR states the defect, the change, and the evidence.
- Before publishing to the public repository, verify docs against the code:
  no fabricated citations, mathematics, or algorithms.

## Where to look for everything else

API documentation is generated from docstrings and published at
https://archimono.github.io/archimono/.

# General behavioral guidelines

Adopted from Andrej Karpathy.

Guidelines to reduce common LLM coding mistakes.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
