"""Tests for simulation runner, telemetry plotting, and CLI execution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from space_manipulator.sim import (
    SimulationTelemetry,
    main,
    plot_comparison,
    plot_telemetry,
    run_simulation,
)


def test_run_simulation_free_floating_and_free_flying() -> None:
    tel_float = run_simulation(
        mode="free-floating",
        duration=4.0,
        target_pos=(1.15, 0.35, 0.25),
        headless=True,
    )
    tel_fly = run_simulation(
        mode="free-flying",
        duration=4.0,
        target_pos=(1.15, 0.35, 0.25),
        headless=True,
    )

    assert isinstance(tel_float, SimulationTelemetry)
    assert isinstance(tel_fly, SimulationTelemetry)
    assert len(tel_float.time) == 2000
    assert tel_float.ee_pos.shape == (2000, 3)
    assert tel_float.base_wrench.shape == (2000, 6)
    assert tel_float.joint_torques.shape == (2000, 3)

    # Both modes converge within 2 cm
    assert tel_float.ee_error_norm[-1] < 0.02
    assert tel_fly.ee_error_norm[-1] < 0.02

    # Free-floating conserves linear & angular momentum and exhibits base reaction
    assert np.allclose(tel_float.base_wrench, 0.0)
    assert np.max(tel_float.linear_momentum_norm) < 1e-6
    assert np.max(tel_float.angular_momentum_norm) < 1e-6
    assert np.max(tel_float.base_pos_drift) > 0.005

    # Free-flying holds base position & attitude tightly compared to free-floating
    assert np.max(tel_fly.base_pos_drift) < 0.005
    assert tel_fly.base_ori_error_deg[-1] < 0.5
    assert np.max(tel_fly.base_ori_error_deg) < 2.0
    assert tel_float.base_ori_error_deg[-1] > 5.0


def test_plot_telemetry_and_comparison(tmp_path: Path) -> None:
    tel_float = run_simulation(mode="free-floating", duration=1.5, headless=True)
    tel_fly = run_simulation(mode="free-flying", duration=1.5, headless=True)

    single_png = plot_telemetry(tel_float, tmp_path / "single.png")
    compare_png = plot_comparison(tel_float, tel_fly, tmp_path / "compare.png")

    for path in (single_png, compare_png):
        assert path.is_file()
        data = path.read_bytes()
        assert len(data) > 1000
        assert data.startswith(b"\x89PNG\r\n\x1a\n")


def test_cli_headless_compare_and_single(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plot_file = tmp_path / "cli_compare.png"
    rc = main(
        [
            "--headless",
            "--mode",
            "compare",
            "--duration",
            "2.0",
            "--plot",
            str(plot_file),
        ]
    )
    assert rc == 0
    assert plot_file.is_file()
    out = capsys.readouterr().out
    assert "[free-floating]" in out
    assert "[free-flying]" in out
    assert "Saved comparison plot" in out

    rc_single = main(
        [
            "--headless",
            "--mode",
            "free-floating",
            "--duration",
            "1.0",
        ]
    )
    assert rc_single == 0
    out_single = capsys.readouterr().out
    assert "[free-floating]" in out_single
