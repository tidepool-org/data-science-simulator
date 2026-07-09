#!/usr/bin/env bash
#
# Rebuild the LoopAlgorithmToPython dynamic library (libLoopAlgorithmToPython.dylib)
# from source, against whichever loop_to_python_api install is currently active
# in this Python environment (pip -e git+... pin, or conda -e ../LoopAlgorithmToPython).
#
# The .dylib checked into the LoopAlgorithmToPython repo can go stale relative to
# its own Swift source (e.g. a new @_cdecl-exported function added but the binary
# never rebuilt), causing confusing `AttributeError: dlsym(...): symbol not found`
# failures at runtime. Run this after installing/updating the swift environment,
# and any time SwiftLoopController-based tests fail with a dlsym/symbol-not-found error.

set -euo pipefail

PKG_ROOT=$(python -c "import os, loop_to_python_api; print(os.path.dirname(os.path.dirname(loop_to_python_api.__file__)))")

if [ ! -f "$PKG_ROOT/Package.swift" ]; then
    echo "error: could not locate the LoopAlgorithmToPython Swift package root (expected Package.swift in $PKG_ROOT)" >&2
    exit 1
fi

echo "Rebuilding LoopAlgorithmToPython dylib in $PKG_ROOT ..."
(
    cd "$PKG_ROOT"
    swift build --configuration release
    cp .build/release/libLoopAlgorithmToPython.dylib loop_to_python_api/libLoopAlgorithmToPython.dylib
)

echo "Done. Rebuilt dylib installed at $PKG_ROOT/loop_to_python_api/libLoopAlgorithmToPython.dylib"
