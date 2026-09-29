from __future__ import annotations

import pytest

from relaygrid.crew.routing import RoadEdge
from relaygrid.planning.temporary_generators import (
    CriticalLoad,
    MobileGenSet,
    REASON_CONNECTOR,
    REASON_EXHAUSTED,
    REASON_FUEL,
    REASON_KVAR,
    REASON_KW,
    REASON_STARTING,
    REASON_UNREACHABLE,
    REASON_VOLTAGE,
    plan_generator_deployment,
)


def road_network():
    return [
        RoadEdge("depot1", "siteA", 10),
        RoadEdge("depot2", "siteA", 30),
        RoadEdge("depot1", "siteB", 20),
        RoadEdge("depot3", "siteC", 15),
    ]


def make_gen(gen_id="G1", node="depot1", kv=12.47, kw=500.0, kvar=200.0,
             connector="camlock", starting=1.0, fuel=1000.0, burn=200.0):
    return MobileGenSet(gen_id, node, kv, kw, kvar, connector, starting, fuel, burn)


def make_load(load_id="L1", node="siteA", kv=12.47, kw=400.0, kvar=150.0,
              connector="camlock", starting=0.8, endurance=4.0, priority=0):
    return CriticalLoad(load_id, node, kv, kw, kvar, connector, starting, endurance, priority)


def test_full_match_assigns_with_travel_and_hookup():
    plan = plan_generator_deployment([make_gen()], [make_load()], road_network())
    assert plan.generator_for("L1") == "G1"
    d = plan.assignments[0]
    assert d.travel_min == 10
    assert d.hookup_min == 60
    assert d.eta_min == 70
    assert d.fuel_endurance_h == 5.0
    assert d.fuel_margin_h == 1.0
    assert plan.uncovered == ()
    assert plan.covered_kw == 400.0


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"kv": 13.8}, REASON_VOLTAGE),
        ({"kw": 300.0}, REASON_KW),
        ({"kvar": 100.0}, REASON_KVAR),
        ({"connector": "bolted"}, REASON_CONNECTOR),
        ({"starting": 0.5}, REASON_STARTING),
        ({"fuel": 400.0}, REASON_FUEL),
    ],
)
def test_incompatible_unit_is_rejected_with_reason(overrides, reason):
    gen = make_gen(**overrides)
    plan = plan_generator_deployment([gen], [make_load()], road_network())
    assert plan.uncovered == ("L1",)
    assert reason in plan.rejections[0].reasons
    assert plan.generator_for("L1") is None


def test_unreachable_site_is_rejected():
    plan = plan_generator_deployment(
        [make_gen()], [make_load(node="siteD")], road_network()
    )
    assert plan.uncovered == ("L1",)
    assert REASON_UNREACHABLE in plan.rejections[0].reasons
    assert plan.rejections[0].nearest_eta_min is None


def test_nearest_eta_reported_when_units_fail_checks_but_reachable():
    plan = plan_generator_deployment(
        [make_gen(kw=300.0)], [make_load()], road_network(), hookup_min=30.0
    )
    r = plan.rejections[0]
    assert REASON_KW in r.reasons
    assert r.nearest_eta_min == 40.0  # 10 travel + 30 hookup


def test_fleet_exhausted_rejects_remaining_load():
    loads = [make_load("L1", priority=1), make_load("L2", priority=0)]
    plan = plan_generator_deployment([make_gen()], loads, road_network())
    assert plan.generator_for("L1") == "G1"
    assert plan.uncovered == ("L2",)
    assert REASON_EXHAUSTED in plan.rejections[0].reasons


def test_no_generator_is_occupied_twice():
    gens = [make_gen("G1", kw=500.0), make_gen("G2", kw=500.0, node="depot2")]
    loads = [make_load("L1", kw=200.0), make_load("L2", kw=200.0, node="siteB")]
    plan = plan_generator_deployment(gens, loads, road_network())
    assigned = [d.gen_id for d in plan.assignments]
    assert len(assigned) == len(set(assigned))
    assert set(assigned) == {"G1", "G2"}


def test_failed_match_does_not_consume_unit():
    # G1 cannot carry the big load at siteA but can serve the small load at siteB
    gens = [make_gen("G1", kw=300.0), make_gen("G2", kw=500.0, node="depot2")]
    loads = [make_load("L1", kw=400.0, node="siteA"),
             make_load("L2", kw=200.0, node="siteB")]
    plan = plan_generator_deployment(gens, loads, road_network())
    assert plan.generator_for("L1") == "G2"
    assert plan.generator_for("L2") == "G1"
    assert plan.uncovered == ()


def test_priority_load_served_first():
    loads = [make_load("L1", priority=0), make_load("L2", priority=10)]
    plan = plan_generator_deployment([make_gen()], loads, road_network())
    assert plan.generator_for("L2") == "G1"
    assert plan.uncovered == ("L1",)


def test_larger_load_first_among_equal_priority():
    loads = [make_load("L1", kw=100.0), make_load("L2", kw=400.0)]
    plan = plan_generator_deployment([make_gen(kw=400.0)], loads, road_network())
    assert plan.generator_for("L2") == "G1"
    assert plan.uncovered == ("L1",)


def test_planning_is_pure_and_idempotent():
    gens = [make_gen("G1"), make_gen("G2", node="depot2")]
    loads = [make_load("L1"), make_load("L2", node="siteB")]
    snapshot_g = [(g.gen_id, g.node, g.kw_rated, g.fuel_l) for g in gens]
    snapshot_l = [(l.load_id, l.node, l.kw) for l in loads]
    first = plan_generator_deployment(gens, loads, road_network())
    second = plan_generator_deployment(gens, loads, road_network())
    assert first == second
    assert [(g.gen_id, g.node, g.kw_rated, g.fuel_l) for g in gens] == snapshot_g
    assert [(l.load_id, l.node, l.kw) for l in loads] == snapshot_l


def test_fuel_reserve_can_reject():
    plan = plan_generator_deployment(
        [make_gen(fuel=1000.0, burn=200.0)], [make_load(endurance=4.0)],
        road_network(), fuel_reserve_h=2.0,
    )
    assert plan.uncovered == ("L1",)
    assert REASON_FUEL in plan.rejections[0].reasons


def test_voltage_tolerance_bridges_nominal_mismatch():
    plan = plan_generator_deployment(
        [make_gen(kv=13.8)], [make_load(kv=12.47)],
        road_network(), voltage_tol_pct=15.0,
    )
    assert plan.generator_for("L1") == "G1"


def test_uncovered_kw_accounting():
    loads = [make_load("L1", kw=400.0), make_load("L2", kw=100.0, node="siteB")]
    plan = plan_generator_deployment([make_gen(kw=400.0)], loads, road_network())
    assert plan.covered_kw == 400.0
    assert plan.uncovered_kw == 100.0


def test_empty_fleet_rejects_everything():
    plan = plan_generator_deployment([], [make_load(), make_load("L2")], road_network())
    assert len(plan.uncovered) == 2
    assert all(REASON_EXHAUSTED in r.reasons for r in plan.rejections)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        plan_generator_deployment([make_gen()], [make_load()], road_network(), hookup_min=-5)
    with pytest.raises(ValueError):
        plan_generator_deployment([make_gen(fuel=-1.0)], [make_load()], road_network())
    with pytest.raises(ValueError):
        plan_generator_deployment([make_gen("G1"), make_gen("G1")], [make_load()], road_network())
