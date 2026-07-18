import subprocess
import sys
import json
from pathlib import Path

from validation.grid import PARAM_GRID
from validation.parity import run as parity_run
from validation.convergence import run as convergence_run
from validation.greeks_check import run as greeks_run
from validation.noarb import run as noarb_run


def test_grid_has_200_points():
    assert len(PARAM_GRID) == 200


def test_parity_gate_passes():
    res = parity_run()
    assert res["passed"], res


def test_greeks_gate_passes():
    res = greeks_run()
    assert res["passed"], res


def test_noarb_gate_passes():
    res = noarb_run()
    assert res["passed"], res


def test_convergence_gate_passes():
    res = convergence_run()
    assert res["passed"], res


def test_runner_writes_json_and_exits_zero():
    proc = subprocess.run([sys.executable, "run_validate.py"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(Path("results/validation.json").read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert len(data["gates"]) == 7
