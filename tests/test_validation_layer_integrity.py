"""TRSET-58: the Pydantic layer must be present, and its absence must be loud."""
import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from tidepool_data_science_simulator.validation import config_validator
from tidepool_data_science_simulator.validation.config_validator import ConfigValidator

REPO = Path(__file__).resolve().parent.parent
VALIDATION = "tidepool_data_science_simulator/validation"


def _reload_blocking(monkeypatch, blocked):
    """Reload config_validator with `blocked` modules unimportable; return module."""
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)
    return importlib.reload(config_validator)


@pytest.fixture
def restore_validator(monkeypatch):
    yield
    monkeypatch.undo()  # unblock modules before reloading
    importlib.reload(config_validator)


def test_validator_constructs_when_layer_present():
    ConfigValidator()


@pytest.mark.parametrize("blocked", [
    ["pydantic"],
    ["tidepool_data_science_simulator.validation.schema_models"],
])
def test_missing_pydantic_layer_raises(monkeypatch, restore_validator, blocked):
    mod = _reload_blocking(monkeypatch, blocked)
    with pytest.raises(RuntimeError, match="pydantic"):
        mod.ConfigValidator()


def test_structurally_invalid_config_is_rejected(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"not_a_scenario": 1}')
    is_valid, errors, _ = ConfigValidator().validate_config_file(str(bad))
    assert not is_valid and errors


def test_validation_modules_are_tracked():
    out = subprocess.run(["git", "ls-files", VALIDATION], cwd=REPO,
                         capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout:
        pytest.skip("not a git checkout")
    tracked = {Path(p).name for p in out.stdout.split()}
    assert {"__init__.py", "schema_models.py",
            "config_validator.py", "value_validators.py"} <= tracked
