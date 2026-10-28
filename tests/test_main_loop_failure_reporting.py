"""
Fault-injection tests for the risk main loop (TRSET-80).

One failure is injected at each stage -- parse, run, metrics, plot, save -- into a small
fixture corpus (tests/fixtures/failure_corpus: 2 TLRs x 2 profile JSONs). ScenarioParserV2
and plot_sim_results are replaced at their boundary with fakes; run_simulations, the
severity summary and the metadata/failures.log writers are the real code. Sims are in-process
fakes, so no Swift sims run.

A JSON's "fault" field selects the injected failure: null (none), "run", "metrics", "save"
or "plot". A truly malformed JSON file injects the parse failure.
"""

import glob
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from tidepool_data_science_simulator.projects.risk import loop_risk_v2_0 as loop_risk

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "failure_corpus")
FAILING_TLR = "TLR-902"
FAILING_JSON = "Simulation-Configuration-TLR-902_median_profile_v1.json"
STAGES = ["parse", "run", "metrics", "plot", "save"]


class _FakeQueue:
    def __init__(self, df):
        self._df = df

    def get(self):
        return self._df


class _FakeSim:
    """In-process stand-in for a Simulation with the attributes run_simulations uses."""

    def __init__(self, sim_id, fault):
        self.sim_id = sim_id
        self.fault = fault
        self.controller = SimpleNamespace()
        n = 288
        df = pd.DataFrame({
            "active": np.ones(n),
            "bg": 110.0 + 20.0 * np.sin(np.linspace(0, 6, n)),
            "bg_sensor": 112.0 + 20.0 * np.sin(np.linspace(0, 6, n)),
            "iob": np.full(n, 1.0),
            "sbr": np.full(n, 0.8),
            "delivered_basal_insulin": np.full(n, 0.07),
            "reported_bolus": np.zeros(n),
            "randint": np.arange(n),
        })
        if fault == "metrics":
            df = df.drop(columns=["iob"])
        self.queue = _FakeQueue(df)

    def start(self):
        if self.fault == "run" and self.sim_id.startswith("pre-noLoop"):
            raise RuntimeError("injected run failure")

    def join(self):
        pass

    def close(self):
        pass

    def get_info_stateless(self):
        if self.fault == "save" and self.sim_id.startswith("pre-noLoop"):
            raise OSError("injected save failure")
        return {"sim_id": self.sim_id}


class _FakeParser:
    def __init__(self, path_to_json_config):
        with open(path_to_json_config) as fh:
            self.fault = json.load(fh)["fault"]  # a malformed JSON raises here: the parse stage
        self.profile = os.path.basename(path_to_json_config).split("_")[1]

    def get_sims(self, override_json_save_dir=None):
        suffix = "t1_{}".format(self.profile)
        ids = ["pre-noLoop_" + suffix, "pre-Loop_NoMitigations_" + suffix, "post-Loop_WithMitigations_" + suffix]
        sims = {}
        for sim_id in ids:
            if self.fault == "plot":
                sim_id = "plotfault-" + sim_id
            sims[sim_id] = _FakeSim(sim_id, self.fault)
        return sims


def _fake_plot(full_results, save=False, save_path=None):
    if any(sim_id.startswith("plotfault-") for sim_id in full_results):
        raise RuntimeError("injected plot failure")
    open(save_path, "w").close()


@pytest.fixture
def run_corpus(tmp_path, monkeypatch):
    """Return run(faults) -> result, where faults maps (tlr, json filename) to a fault name."""
    monkeypatch.setattr(loop_risk, "ScenarioParserV2", _FakeParser)
    monkeypatch.setattr(loop_risk, "plot_sim_results", _fake_plot)

    def run(faults=None):
        corpus = tmp_path / "corpus"
        shutil.rmtree(corpus, ignore_errors=True)
        shutil.copytree(CORPUS_DIR, corpus)
        for (tlr, name), fault in (faults or {}).items():
            path = corpus / tlr / name
            path.write_text('{"broken": ' if fault == "parse" else json.dumps({"fault": fault}))

        results_dir = tmp_path / "results"
        results_dir.mkdir(exist_ok=True)
        exit_code = loop_risk.main(scenarios_dir=str(corpus), results_dir=str(results_dir), num_procs=2)

        run_dir = glob.glob(str(results_dir / "Risk_Run_*"))[-1]
        risk_csvs = glob.glob(os.path.join(run_dir, "Risk_Results_*.csv"))
        with open(os.path.join(run_dir, "metadata.json")) as fh:
            raw = fh.read()
        return SimpleNamespace(
            exit_code=exit_code,
            run_dir=run_dir,
            metadata=json.loads(raw),
            metadata_keys=list(json.loads(raw).keys()),
            failures_log=open(os.path.join(run_dir, "failures.log")).read(),
            risk_df=pd.read_csv(risk_csvs[0]) if risk_csvs else None,
            n_risk_csvs=len(risk_csvs),
        )

    return run


def _sims_with_tsv(run_dir, tlr):
    return {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(run_dir, tlr, "*.tsv"))}


def test_clean_run_reports_nothing_failed(run_corpus):
    result = run_corpus()

    assert result.exit_code == 0
    assert result.metadata["failures"] == []
    assert result.metadata["scenarios_attempted"] == 4
    assert result.metadata["scenarios_succeeded"] == 4
    assert len(result.risk_df) == 12
    assert "0 failure(s)" in result.failures_log


@pytest.mark.parametrize("stage", STAGES)
def test_one_failure_at_each_stage(run_corpus, stage):
    result = run_corpus({(FAILING_TLR, FAILING_JSON): stage})

    # (a) the other scenarios complete
    assert len(_sims_with_tsv(result.run_dir, "TLR-901")) == 6
    other = _sims_with_tsv(result.run_dir, FAILING_TLR)
    assert {"pre-noLoop_t1_adolescent", "post-Loop_WithMitigations_t1_adolescent"} <= other
    assert result.metadata["scenarios_attempted"] == 4
    assert result.metadata["scenarios_succeeded"] == 3

    # (b) no row for the failed JSON, no dummy or NaN rows
    assert "dummy" not in result.risk_df.columns
    assert len(result.risk_df) == 9
    assert not result.risk_df["lbgi"].isna().any()
    assert not result.risk_df["scenario_name"].eq(FAILING_JSON).any()

    # (c) recorded in metadata.json and failures.log with the right stage
    failures = result.metadata["failures"]
    assert failures and {f["stage"] for f in failures} == {stage}
    assert all(f["tlr"] == FAILING_TLR and f["json"] == FAILING_JSON for f in failures)
    assert all(f["exception_type"] and f["message"] for f in failures)
    if stage in ("run", "metrics", "save"):
        assert all(f["sim_id"] for f in failures)
    else:
        assert all(f["sim_id"] is None for f in failures)
    assert "Traceback" in result.failures_log
    assert f"stage: {stage}" in result.failures_log

    # (d) exit code: some failures, partial results
    assert result.exit_code == 1


def test_metadata_key_order(run_corpus):
    result = run_corpus({(FAILING_TLR, FAILING_JSON): "run"})

    assert result.metadata_keys[:3] == ["timestamp", "scenarios_attempted", "scenarios_succeeded"]
    assert result.metadata_keys[-2:] == ["failures_log", "failures"]
    assert result.metadata["failures_log"] == "failures.log"


def test_metadata_failures_key_present_and_last_on_clean_run(run_corpus):
    result = run_corpus()

    assert result.metadata_keys[-1] == "failures"


def test_every_scenario_fails(run_corpus):
    faults = {(tlr, f"Simulation-Configuration-{tlr}_{p}_profile_v1.json"): "run"
              for tlr in ("TLR-901", "TLR-902") for p in ("median", "adolescent")}
    result = run_corpus(faults)

    assert result.n_risk_csvs == 0
    assert result.exit_code == 2
    assert result.metadata["scenarios_succeeded"] == 0
    assert result.metadata["scenarios_attempted"] == 4
    assert len({(f["tlr"], f["json"]) for f in result.metadata["failures"]}) == 4


def test_one_failed_profile_skips_only_its_tlr_rtf(run_corpus):
    result = run_corpus({(FAILING_TLR, FAILING_JSON): "metrics"})

    assert result.exit_code == 1
    assert glob.glob(os.path.join(result.run_dir, "TLR-901", "*.rtf"))
    assert not glob.glob(os.path.join(result.run_dir, FAILING_TLR, "*.rtf"))
    skipped = result.metadata["severity_summaries_skipped"]
    assert skipped == [{"tlr": FAILING_TLR, "reason": "incomplete TLR: 1 failed"}]
    # run-level fields stay between the timestamp block and the failures block
    assert result.metadata_keys[-1] == "failures"


def test_malformed_json_does_not_stop_later_scenarios(run_corpus):
    """The first-listed JSON of the corpus being malformed used to end the whole run."""
    faults = {(tlr, f"Simulation-Configuration-{tlr}_{p}_profile_v1.json"): "parse"
              for tlr in ("TLR-901",) for p in ("median",)}
    result = run_corpus(faults)

    assert result.metadata["scenarios_succeeded"] == 3
    assert len(result.risk_df) == 9
    assert result.exit_code == 1
