"""Fast tests for the ``enumeration.get`` backend selector."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from archimono.assignment import enumeration
from archimono.assignment.enumeration import DEFAULT_BACKEND
from archimono.assignment.enumeration.icet import IcetEnumerator
from archimono.assignment.enumeration.orderly import OrderlyEnumerator
from archimono.tilings import registry


class TestBackendSelector:
    """``enumeration.get`` resolves backend names to enumerator instances."""

    def test_default_is_orderly(self) -> None:
        """No argument selects the orderly backend."""
        assert isinstance(enumeration.get(), OrderlyEnumerator)

    def test_default_backend_constant(self) -> None:
        """DEFAULT_BACKEND names the orderly backend."""
        assert DEFAULT_BACKEND == "orderly"
        assert isinstance(enumeration.get(DEFAULT_BACKEND), OrderlyEnumerator)

    def test_orderly_selection(self) -> None:
        """``"orderly"`` selects OrderlyEnumerator."""
        assert isinstance(enumeration.get("orderly"), OrderlyEnumerator)

    def test_icet_selection(self) -> None:
        """``"icet"`` selects IcetEnumerator."""
        assert isinstance(enumeration.get("icet"), IcetEnumerator)

    def test_unknown_backend_raises(self) -> None:
        """An unrecognised backend name raises KeyError listing the options."""
        with pytest.raises(
            KeyError, match=r"Unknown enumeration backend.*Available:.*orderly"
        ):
            enumeration.get("nonexistent")

    def test_returns_fresh_instances(self) -> None:
        """Each call constructs a new enumerator instance."""
        assert enumeration.get() is not enumeration.get()


class TestBackendInterchangeability:
    """Both backends accept an identical ``enumerate`` call signature."""

    def test_identical_kwargs_agree(self) -> None:
        """get(b).enumerate(...) accepts the same kwargs and agrees for both b."""
        tiling = registry.get("triangular")
        kwargs = {"n_b": 2, "supercell": (2, 2), "min_cut": 0, "override": False}

        results = {
            backend: enumeration.get(backend).enumerate(tiling, **kwargs)
            for backend in ("orderly", "icet")
        }

        assert len(results["orderly"]) == len(results["icet"]) > 0


class TestIcetFreeImport:
    """The default orderly path must work without the optional icet library."""

    def test_import_and_enumerate_without_icet(self) -> None:
        """Importing enumeration and running orderly succeeds with icet absent.

        Runs in a subprocess with ``icet`` blocked in ``sys.modules`` so the
        check is isolated from the dev environment (which has icet installed).
        Guards both the icet-free value proposition and against a future eager
        ``icet`` import creeping back into the import chain.
        """
        script = textwrap.dedent(
            """
            import sys
            # Make `import icet` fail, simulating it not being installed.
            sys.modules["icet"] = None
            sys.modules["icet.tools"] = None

            from archimono.assignment import enumeration
            from archimono.tilings import registry

            enum = enumeration.get()
            assert type(enum).__name__ == "OrderlyEnumerator"
            configs = enum.enumerate(
                registry.get("triangular"), n_b=2, supercell=(2, 2)
            )
            assert len(configs) > 0
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
