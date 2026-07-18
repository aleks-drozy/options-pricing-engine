import subprocess
import sys
import json
from pathlib import Path

from validation.grid import PARAM_GRID
from validation.parity import MC_SAMPLE_STRIDE, run as parity_run
from validation.convergence import TREE_SAMPLE_STRIDE, run as convergence_run
from validation.greeks_check import GREEKS_SAMPLE_STRIDE, run as greeks_run
from validation.noarb import NOARB_SAMPLE_STRIDE, run as noarb_run


def test_grid_has_200_points():
    assert len(PARAM_GRID) == 200


def test_gate_sampling_covers_nonzero_r_and_q():
    # Regression for the stride-aliasing bug: PARAM_GRID's innermost (r, q)
    # axis has period 5, so any stride that is a multiple of 5 only ever
    # samples r=0, q=0 points. Gate strides must stay coprime with 5.
    strides = {TREE_SAMPLE_STRIDE, NOARB_SAMPLE_STRIDE,
               MC_SAMPLE_STRIDE, GREEKS_SAMPLE_STRIDE}
    for stride in strides:
        assert stride % 5 != 0, f"stride {stride} aliases the r/q grid axis"
        sub = PARAM_GRID[::stride]
        assert any(g["r"] > 0 for g in sub), f"stride {stride} samples no r>0"
        assert any(g["q"] > 0 for g in sub), f"stride {stride} samples no q>0"


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
