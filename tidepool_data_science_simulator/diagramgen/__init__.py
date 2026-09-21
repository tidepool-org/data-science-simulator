"""
Trace-based architecture diagram generator (TRSET-51).

Instruments a real, in-process simulation run and emits Mermaid figures that are
*evidence of executed behavior* rather than hand-maintained assertions about
permitted structure.

The package is a pure consumer of the simulator: nothing here modifies
``Simulation``, ``run_simulations()``, the controllers, the patient or the
parser. See ``README.md`` in this directory for the execution model and the
rationale behind the capture mechanism.
"""

__all__ = ["__version__"]

# Bumped when the *shape* of an emitted artifact changes, so a diff in a
# committed figure can be attributed to a generator change rather than to an
# architecture change.
__version__ = "1.0.0"
