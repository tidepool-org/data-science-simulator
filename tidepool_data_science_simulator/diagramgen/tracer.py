"""
Runtime capture of cross-package calls during a real simulation.

Two mechanisms, because neither one alone sees the whole picture:

``sys.setprofile``
    The workhorse. Reports every Python call and return, giving both the callee
    frame and -- via ``frame.f_back`` -- the caller, so an edge is observed with
    its concrete bindings on both ends. Filtering happens *inside* the callback
    against the allowlist rather than post-hoc over a full trace, which avoids a
    second traversal of roughly 1,100 control cycles.

``sys.monitoring`` (PEP 669, Python 3.12+), narrowly scoped
    The ticket assumed ``setprofile``'s ``c_call`` event would expose the
    ``ctypes.CDLL`` -> ``libLoopAlgorithmToPython.dylib`` boundary. Measured
    against a real run, it does not: ``swift_lib.getLoopRecommendations`` is a
    ctypes ``_FuncPtr`` (``PyCFuncPtr``), not a ``PyCFunction``, so CPython emits
    no ``c_call`` for it -- the only ``_ctypes`` events on the wire come from
    ``POINTER`` during ``restype`` setup. ``sys.monitoring``'s ``CALL`` event
    *does* report it. A ``CALL`` callback returning ``sys.monitoring.DISABLE``
    for every non-ctypes callable turns the event off permanently at that
    bytecode offset, so after one pass the probe costs nothing at the call sites
    that are not native boundaries.

Nothing in this module writes to the simulator's results directory, and no
argument or return **value** is ever recorded -- only call identity and argument
*shape* (module, qualname, resolved class, arity, type names). No glucose array
and no LBGI/DKAI scalar reaches disk.
"""

import os
import sys
import time

from tidepool_data_science_simulator.diagramgen.naming import NATIVE_PACKAGE, Node, top_package

__all__ = ["CallRecord", "CallTracer", "PHASE_CONSTRUCTION", "PHASE_METRICS", "PHASE_RUN"]

PHASE_CONSTRUCTION = "construction"
PHASE_RUN = "run"
PHASE_METRICS = "metrics"

# Locals whose presence marks an instance method, used to resolve the concrete
# class standing behind a call site typed against a base class.
_SELF = "self"


class CallRecord(object):
    """One observed call. Identity and shape only -- never values."""

    __slots__ = (
        "seq", "stage", "phase", "timestep", "kind",
        "caller", "callee", "callee_qualname", "caller_qualname",
        "arity", "arg_types", "ts_ns", "pid", "caller_attributed",
    )

    def __init__(self, seq, stage, phase, timestep, kind, caller, caller_qualname,
                 callee, callee_qualname, arity, arg_types, ts_ns, pid,
                 caller_attributed=False):
        self.seq = seq
        self.stage = stage
        self.phase = phase
        self.timestep = timestep
        self.kind = kind
        self.caller = caller
        self.caller_qualname = caller_qualname
        self.callee = callee
        self.callee_qualname = callee_qualname
        self.arity = arity
        self.arg_types = arg_types
        self.ts_ns = ts_ns
        self.pid = pid
        self.caller_attributed = caller_attributed

    @property
    def is_cross_package(self):
        return self.caller.package != self.callee.package

    def to_json_dict(self):
        """Serialisable form for ``trace.jsonl``.

        Timestamps, durations and PIDs appear here and nowhere else -- they are
        deliberately kept out of every rendered artifact so the drift gate is
        not fired by wall-clock noise.
        """
        return {
            "seq": self.seq,
            "stage": self.stage,
            "phase": self.phase,
            "timestep": self.timestep,
            "kind": self.kind,
            "caller": {
                "package": self.caller.package,
                "module": self.caller.module,
                "resolved_class": self.caller.component or None,
                "qualname": self.caller_qualname,
                "attributed": self.caller_attributed,
            },
            "callee": {
                "package": self.callee.package,
                "module": self.callee.module,
                "resolved_class": self.callee.component or None,
                "qualname": self.callee_qualname,
            },
            "arity": self.arity,
            "arg_types": list(self.arg_types),
            "ts_ns": self.ts_ns,
            "pid": self.pid,
        }


def _resolved_class(frame):
    """Return the concrete class bound to ``self`` in ``frame``, or ``""``.

    ``co_varnames`` is checked first so ``f_locals`` -- which materialises a
    snapshot dict -- is only touched for frames that can actually carry one.
    """
    code = frame.f_code
    if code.co_argcount == 0 or code.co_varnames[:1] != (_SELF,):
        return ""
    instance = frame.f_locals.get(_SELF)
    if instance is None:
        return ""
    try:
        return type(instance).__name__
    except Exception:  # pragma: no cover - defensive; type() on an exotic proxy
        return ""


def _arg_shape(frame):
    """Return ``(arity, [type names])`` for a frame's positional arguments.

    ``self`` is dropped: it is already carried as the resolved class, and its
    type name would be redundant on every row.
    """
    code = frame.f_code
    names = code.co_varnames[: code.co_argcount]
    if names[:1] == (_SELF,):
        names = names[1:]
    locals_snapshot = frame.f_locals
    types = []
    for name in names:
        try:
            types.append(type(locals_snapshot[name]).__name__)
        except KeyError:
            # A generator frame can be entered before every argument is bound.
            types.append("unbound")
    return len(names), types


class CallTracer(object):
    """Captures allowlisted calls for the duration of a ``with`` block.

    The tracer is stateful across the whole generation run: ``stage`` and
    ``phase`` are set by the runner as it moves from building simulations to
    executing them to computing metrics, and the timestep counter advances on
    the allowlisted boundary call so the sequence diagram can be cut from a
    single control cycle.
    """

    def __init__(self, allowlist):
        self.allowlist = allowlist
        self.records = []
        # Resolved absolute paths of every ``reusable.*`` pointer file loaded,
        # captured by observing the parser's return locals rather than by
        # modifying the parser.
        self.pointer_paths = []
        # (module, qualname) of every allowlisted-package function that actually
        # executed. The static pass uses this to tell "this call site sits in a
        # function that never ran" apart from "the function ran but did not take
        # this branch" -- a distinction the coverage report reports separately.
        self.executed_functions = set()

        self.stage = None
        self.phase = PHASE_CONSTRUCTION
        self.timestep = -1
        # When the generator itself stands in for a real runtime call site -- it
        # does for exactly one edge, the results-path metrics call that
        # ``run_simulations()`` makes -- the caller is recorded as that call site
        # and every record carries ``attributed: true`` to say so.
        self.attributed_caller = None
        self.attributed_caller_qualname = ""

        self._prefixes = allowlist.module_prefixes
        self._boundary = allowlist.timestep_boundary
        self._pointer_capture = allowlist.pointer_capture
        self._seq = 0
        self._pid = os.getpid()
        self._module_by_filename = {}
        self._funcptr_types = {}
        self._monitoring_tool_id = None
        self._previous_profiler = None
        self._own_files = frozenset(self._own_module_files())

    # -- lifecycle ---------------------------------------------------------

    def __enter__(self):
        self._install()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._remove()
        return False

    def _own_module_files(self):
        """Files belonging to the generator itself, so it never traces itself."""
        here = os.path.dirname(os.path.abspath(__file__))
        return [os.path.join(here, name) for name in os.listdir(here) if name.endswith(".py")]

    def _install(self):
        self._index_ctypes_libraries()
        self._install_native_probe()
        self._previous_profiler = sys.getprofile()
        sys.setprofile(self._on_profile)

    def _remove(self):
        sys.setprofile(self._previous_profiler)
        self._previous_profiler = None
        self._remove_native_probe()

    # -- native (ctypes) probe --------------------------------------------

    def _index_ctypes_libraries(self):
        """Map each loaded ``CDLL``'s private ``_FuncPtr`` type to its basename.

        ``ctypes.CDLL`` builds a fresh ``_FuncPtr`` subclass per library, so the
        type of a function pointer identifies its library exactly -- no guessing
        from the symbol name. Only the basename is kept: the dylib lives under
        an editable install path that must not reach a figure.
        """
        import ctypes

        self._funcptr_types = {}
        for module in list(sys.modules.values()):
            if module is None or top_package(getattr(module, "__name__", "")) not in self._prefixes:
                continue
            for value in list(vars(module).values()):
                if isinstance(value, ctypes.CDLL):
                    func_ptr_type = getattr(value, "_FuncPtr", None)
                    if func_ptr_type is not None:
                        self._funcptr_types[func_ptr_type] = os.path.basename(value._name or "unknown")

    def _install_native_probe(self):
        if not self._funcptr_types:
            return
        monitoring = sys.monitoring
        for candidate in range(monitoring.DEBUGGER_ID, 6):
            # PROFILER_ID is claimed by sys.setprofile; never contend for it.
            if candidate == monitoring.PROFILER_ID:
                continue
            if monitoring.get_tool(candidate) is None:
                self._monitoring_tool_id = candidate
                break
        if self._monitoring_tool_id is None:  # pragma: no cover - all ids taken
            return
        monitoring.use_tool_id(self._monitoring_tool_id, "trset51-diagramgen")
        monitoring.register_callback(self._monitoring_tool_id, monitoring.events.CALL, self._on_native_call)
        monitoring.set_events(self._monitoring_tool_id, monitoring.events.CALL)

    def _remove_native_probe(self):
        if self._monitoring_tool_id is None:
            return
        monitoring = sys.monitoring
        monitoring.set_events(self._monitoring_tool_id, 0)
        monitoring.register_callback(self._monitoring_tool_id, monitoring.events.CALL, None)
        monitoring.free_tool_id(self._monitoring_tool_id)
        self._monitoring_tool_id = None

    def _on_native_call(self, code, instruction_offset, callable_, arg0):
        """``sys.monitoring`` CALL callback, scoped to ctypes function pointers.

        Returning ``DISABLE`` retires the event at this exact bytecode offset,
        so the probe converges to firing only at genuine native call sites.
        """
        library = self._funcptr_types.get(type(callable_))
        if library is None:
            return sys.monitoring.DISABLE

        caller_module = self._module_for_filename(code.co_filename)
        caller = Node(top_package(caller_module), caller_module, "")
        callee = Node(NATIVE_PACKAGE, library, "")
        symbol = getattr(callable_, "__name__", "unknown")
        self._append(
            kind="native_call",
            caller=caller,
            caller_qualname=code.co_qualname,
            callee=callee,
            callee_qualname=symbol,
            arity=0,
            arg_types=[],
        )
        return None

    # -- profile callback --------------------------------------------------

    def _on_profile(self, frame, event, arg):
        if event == "call":
            self._on_python_call(frame)
        elif event == "return":
            self._on_python_return(frame)
        return None

    def _on_python_call(self, frame):
        code = frame.f_code
        qualname = code.co_qualname

        # Run structure first: the boundary call delimits control cycles and is
        # bookkeeping, not an edge.
        if qualname == self._boundary and self.phase == PHASE_RUN:
            self.timestep += 1

        # Module-level code is import execution, not a call between components:
        # importing `tidepool_data_science_metrics.glucose.glucose` would
        # otherwise show up as an edge from `importlib._bootstrap`.
        if qualname == "<module>":
            return

        module = frame.f_globals.get("__name__", "")
        if top_package(module) not in self._prefixes:
            return

        self.executed_functions.add((module, qualname))

        back = frame.f_back
        if back is None:
            return

        caller, caller_qualname, attributed = self._caller_for(back)
        if caller is None:
            return

        callee_package = top_package(module)
        is_cross_package = caller.package != callee_package
        if not is_cross_package and not self.allowlist.matches_named_step(qualname):
            return

        arity, arg_types = _arg_shape(frame)
        self._append(
            kind="call",
            caller=caller,
            caller_qualname=caller_qualname,
            callee=Node(callee_package, module, _resolved_class(frame)),
            callee_qualname=qualname,
            arity=arity,
            arg_types=arg_types,
            caller_attributed=attributed,
        )

    def _caller_for(self, back):
        """Resolve the calling side of an edge.

        The generator drives the run, so its own frames are not architecture.
        They are dropped, except where an attributed caller has been declared
        for the current phase -- the generator then stands in for the real
        runtime call site and the record is flagged accordingly.
        """
        if back.f_code.co_filename in self._own_files:
            if self.attributed_caller is None:
                return None, "", False
            return self.attributed_caller, self.attributed_caller_qualname, True
        caller_module = back.f_globals.get("__name__", "")
        return (
            Node(top_package(caller_module), caller_module, _resolved_class(back)),
            back.f_code.co_qualname,
            False,
        )

    def _on_python_return(self, frame):
        """Capture resolved ``reusable.*`` pointer paths as the parser returns.

        The parser searches several candidate directories and returns the loaded
        object, not the path it came from, so the reference scenario's pointer
        ``base_median_2_0_v1`` -- which exists under both ``simulations/base/``
        and ``simulations/base_urai/`` -- is not identified by the scenario
        filename alone. Reading the returning frame's locals recovers the path
        that actually hit, with no change to the parser.
        """
        if frame.f_code.co_qualname != self._pointer_capture:
            return
        locals_snapshot = frame.f_locals
        for name in ("json_path", "csv_path"):
            path = locals_snapshot.get(name)
            if isinstance(path, str) and os.path.isfile(path):
                # The parser builds these by joining a `..`-relative package
                # directory; normalise so the recorded provenance is a single
                # unambiguous location.
                self.pointer_paths.append(os.path.normpath(os.path.abspath(path)))
                return

    # -- recording ---------------------------------------------------------

    def _append(self, kind, caller, caller_qualname, callee, callee_qualname, arity,
                arg_types, caller_attributed=False):
        self.records.append(
            CallRecord(
                seq=self._seq,
                stage=self.stage,
                phase=self.phase,
                timestep=self.timestep if self.phase == PHASE_RUN else None,
                kind=kind,
                caller=caller,
                caller_qualname=caller_qualname,
                callee=callee,
                callee_qualname=callee_qualname,
                arity=arity,
                arg_types=arg_types,
                ts_ns=time.perf_counter_ns(),
                pid=self._pid,
                caller_attributed=caller_attributed,
            )
        )
        self._seq += 1

    def _module_for_filename(self, filename):
        """Best-effort dotted module name for a code object's file."""
        cached = self._module_by_filename.get(filename)
        if cached is not None:
            return cached
        resolved = ""
        for name, module in list(sys.modules.items()):
            if getattr(module, "__file__", None) == filename:
                resolved = name
                break
        if not resolved:
            resolved = os.path.splitext(os.path.basename(filename))[0]
        self._module_by_filename[filename] = resolved
        return resolved

    # -- run structure -----------------------------------------------------

    def begin_stage(self, stage, phase, attributed_caller=None, attributed_caller_qualname=""):
        """Mark the start of a stage/phase; resets the control-cycle counter."""
        self.stage = stage
        self.phase = phase
        self.timestep = -1
        self.attributed_caller = attributed_caller
        self.attributed_caller_qualname = attributed_caller_qualname
