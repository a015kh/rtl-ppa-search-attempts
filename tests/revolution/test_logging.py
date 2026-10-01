from revolution.algorithm import Heuristic
from revolution.logging import EoHLogger


def _candidate(candidate_id, score, metrics):
    candidate = Heuristic("thought", "code", "")
    candidate.id = candidate_id
    candidate.score = score
    candidate.status = "success"
    candidate.ppa_success = True
    candidate.ppa_metrics = metrics
    candidate.code_file_path = f"/tmp/{candidate_id}.sv"
    return candidate


def test_best_record_constraint_filter_excludes_timing_violation(tmp_path):
    logger = EoHLogger(
        "prob",
        "bench",
        "model",
        str(tmp_path),
        None,
        best_record_require_constraints=True,
    )
    violating = _candidate(
        "violating",
        2.0,
        {
            "clock_ns": 1.0,
            "eff_clk_period": 1.2,
            "constraint_met": False,
            "area": 5.0,
            "power": 5.0,
        },
    )
    feasible = _candidate(
        "feasible",
        1.0,
        {
            "clock_ns": 1.0,
            "eff_clk_period": 0.9,
            "constraint_met": True,
            "area": 10.0,
            "power": 10.0,
        },
    )

    stats = logger._calculate_ppa_stats([violating, feasible])

    assert stats["best_score"] == 1.0
    assert stats["best_metrics"]["constraint_met"] is True
    assert stats["record_candidate_count"] == 1
    assert stats["constraint_filtered_count"] == 1


def test_pareto_front_spans_clocks_and_removes_dominated_points(tmp_path):
    logger = EoHLogger(
        "prob",
        "bench",
        "model",
        str(tmp_path),
        None,
        best_record_require_constraints=True,
    )
    logger.all_ppa_records = [
        {
            "id": "candidate-a",
            "generation": 1,
            "strategy": "M-I",
            "code_file_path": "/tmp/a.sv",
            "score": 1.0,
            "ppa_metrics": {
                "clock_results": [
                    {
                        "success": True,
                        "metrics": {
                            "clock_ns": 1.0,
                            "area": 20.0,
                            "power": 8.0,
                            "eff_clk_period": 0.9,
                            "constraint_met": True,
                        },
                    },
                    {
                        "success": True,
                        "metrics": {
                            "clock_ns": 3.0,
                            "area": 10.0,
                            "power": 4.0,
                            "eff_clk_period": 2.8,
                            "constraint_met": True,
                        },
                    },
                ]
            },
        },
        {
            "id": "candidate-b",
            "generation": 2,
            "strategy": "M-E",
            "code_file_path": "/tmp/b.sv",
            "score": 0.5,
            "ppa_metrics": {
                "clock_results": [
                    {
                        "success": True,
                        "metrics": {
                            "clock_ns": 3.0,
                            "area": 12.0,
                            "power": 5.0,
                            "eff_clk_period": 2.9,
                            "constraint_met": True,
                        },
                    },
                    {
                        "success": True,
                        "metrics": {
                            "clock_ns": 1.0,
                            "area": 1.0,
                            "power": 1.0,
                            "eff_clk_period": 1.2,
                            "constraint_met": False,
                        },
                    },
                ]
            },
        },
    ]

    front = logger._pareto_front()

    assert [(point["clock_ns"], point["area"], point["power"]) for point in front] == [
        (1.0, 20.0, 8.0),
        (3.0, 10.0, 4.0),
    ]
