# Contributing to archimono

Thanks for your interest in contributing! This guide covers what you need to
get a change merged. If you use an AI assistant, the same rules apply to its
work, and agents have a few extra instructions in [CLAUDE.md](CLAUDE.md).

## Development setup

```bash
git clone https://github.com/archimono/archimono
cd archimono
pip install -e ".[dev]"
```

## Running the tests

The tests come in three tiers, from quick checks to exhaustive ones:

```bash
pytest -m fast       # small-graph unit tests, for use while editing
pytest -m contract   # CLI and multi-tiling integration tests, before each commit
pytest -m heavy      # large supercells and exhaustive checks, before each PR
pytest -m ""         # all three tiers
```

Running plain `pytest` covers the fast and contract tiers.

## Code style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html),
and Ruff enforces the `E`, `F`, `W`, `I`, `UP`, and `ANN` rules at a line length
of 88. All function signatures need full type annotations so that
`mypy --strict` passes.

**Docstrings.** Give every public module, class, and function a Google-style
docstring. A one-line summary is enough when the name and types already explain
the rest, so add `Args:`, `Returns:`, or `Raises:` only when they tell the reader
something more. Tests usually don't need a docstring unless their purpose isn't
clear from the name and assertions.

Keep any assumption or limit a caller needs in the docstring, because the
published documentation is generated from docstrings.

**Comments.** Use comments for what the code can't show: invariants, why the
arithmetic must be exact, complexity, constraints, and warnings. There's no need
to describe what a line does.

**Citations.** If an algorithm or result comes from a paper, textbook, or other
source, cite it in the docstring or an inline comment with the author, title,
year, and equation or section number:

```python
# Hadlock (1975), "Finding a Maximum Cut of a Planar Graph in Polynomial Time",
# SIAM J. Comput. 4(3), pp. 221–225. https://doi.org/10.1137/0204019
# §§3–4: reduce MAX-CUT to minimum odd-vertex pairing in the dual.
```

Readers need to be able to open what you cite, so don't cite a gitignored
directory, a scratch path, or an unmerged branch.

**Error messages.** Library code should name its own parameters in errors. CLI
flags and other CLI wording belong in `src/archimono/cli/`.

## Submitting changes

1. Create a branch off `main`.
2. Keep the change focused, and add or update tests for any change in behaviour.
3. Make sure these all pass:

   ```bash
   ruff check src/ scripts/ tests/
   mypy src/
   pytest -m ""
   ```

4. Open a pull request that explains what changed and why.

## Documentation

The API reference at https://archimono.github.io/archimono/ is generated from
docstrings. Build it locally with `sphinx-build -b html docs docs/_build/html`
after `pip install -e ".[docs]"`.
