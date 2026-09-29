from relaygrid.crew.routing import RoadEdge
from relaygrid.planning.generator_siting import (
    ACTIVE_POWER,
    ALREADY_OCCUPIED,
    APPARENT_POWER,
    CONNECTOR_MISSING,
    FUEL_ENDURANCE,
    GeneratorInventory,
    NO_ROUTE,
    READY_DEADLINE,
    STARTING_CURRENT,
    VOLTAGE_MISMATCH,
    CriticalLoad,
    GeneratorUnit,
    apparent_kva,
    evaluate_unit,
    operating_current_a,
    plan_generators,
)

PADS = frozenset({"camlock", "l21-30"})


def g(unit_id="g1", location="yard", voltage_v=400.0, rated_kw=100.0, rated_kva=125.0,
      sustained_a=180.0, inrush_a=360.0, connectors=PADS, tank_l=500.0, burn_lph=10.0,
      hookup_min=30.0):
    return GeneratorUnit(unit_id, location, voltage_v, rated_kw, rated_kva,
                         sustained_a, inrush_a, connectors, tank_l, burn_lph, hookup_min)


def load(load_id="l1", node="site", voltage_v=400.0, active_kw=60.0, reactive_kvar=30.0,
         connector="camlock", starting_a=200.0, priority=1, required_hours=24.0,
         required_ready_min=None, phases=3):
    return CriticalLoad(load_id, node, voltage_v, active_kw, reactive_kvar, connector,
                        starting_a, priority, required_hours, required_ready_min, phases)


def edges():
    return [RoadEdge("yard", "site", 45), RoadEdge("site", "shed", 15)]


def test_apparent_and_current_three_phase():
    assert abs(apparent_kva(load(active_kw=60, reactive_kvar=30)) - 67.0820393249937) < 1e-9
    iop = operating_current_a(load(active_kw=60, reactive_kvar=30))
    expected = apparent_kva(load(active_kw=60, reactive_kvar=30)) * 1000 / (3 ** 0.5 * 400)
    assert abs(iop - expected) < 1e-9


def test_single_feasible_assignment_commits_and_reserves():
    stock = GeneratorInventory([g()])
    plan = plan_generators([load()], stock, edges())
    assert plan.committed is True
    p = plan.assignments[0]
    assert p.unit_id == "g1"
    assert p.travel_min == 45 and p.hookup_min == 30 and p.ready_at_min == 75
    assert p.spare_kw == 40
    assert stock.available_ids == frozenset()  # committed occupancy


def test_unit_not_reused_within_batch():
    stock = GeneratorInventory([g()])
    l2 = load("l2")
    plan = plan_generators([load("l1"), l2], stock, edges())
    assert plan.committed is False
    assert len(plan.assignments) == 1
    assert plan.assignments[0].load_id == "l1"
    rej = plan.uncovered[0].rejections[0]
    assert rej.unit_id == "g1" and rej.reasons == (ALREADY_OCCUPIED,)
    # rollback: partial failure leaves inventory untouched
    assert stock.available_ids == frozenset({"g1"})


def test_two_units_cover_two_loads():
    stock = GeneratorInventory([g("g1"), g("g2", rated_kw=150, rated_kva=175)])
    plan = plan_generators([load("l1"), load("l2", active_kw=120, starting_a=300)],
                           stock, edges())
    assert plan.committed is True
    assert {p.load_id: p.unit_id for p in plan.assignments} == {"l1": "g1", "l2": "g2"}
    assert stock.available_ids == frozenset()


def test_smallest_feasible_unit_is_chosen():
    stock = GeneratorInventory([
        g("big", rated_kw=300, rated_kva=350),
        g("small", rated_kw=100, rated_kva=125),
    ])
    plan = plan_generators([load()], stock, edges())
    assert plan.assignments[0].unit_id == "small"


def test_priority_order_then_id():
    stock = GeneratorInventory([g()])  # only one unit fits the low load
    high = load("high", active_kw=95, priority=10)
    low = load("low", active_kw=10, priority=1)
    plan = plan_generators([low, high], stock, edges())
    assert plan.assignments[0].load_id == "high"
    assert plan.uncovered[0].load_id == "low"


def test_voltage_mismatch_and_connector_and_power_reasons():
    unit = g(voltage_v=480, rated_kw=10, rated_kva=12, connectors=frozenset({"l21-30"}),
             sustained_a=10, inrush_a=10, tank_l=10, burn_lph=50)
    _, _, _, reasons = evaluate_unit(load(), unit, edges())
    assert VOLTAGE_MISMATCH in reasons
    assert ACTIVE_POWER in reasons
    assert APPARENT_POWER in reasons
    assert STARTING_CURRENT in reasons
    assert (reasons[0],) == (VOLTAGE_MISMATCH,)  # canonical order
    assert CONNECTOR_MISSING in reasons
    assert FUEL_ENDURANCE in reasons
    # no route: unit sits at a disconnected staging point
    _, _, _, no_route = evaluate_unit(load(required_ready_min=10),
                                      g(location="island"), edges())
    assert NO_ROUTE in no_route and READY_DEADLINE not in no_route


def test_ready_deadline_combines_travel_and_hookup():
    unit = g(hookup_min=30)  # travel 45 + hookup 30 = 75
    _, ready_at, _, late = evaluate_unit(load(required_ready_min=60), unit, edges())
    assert ready_at == 75 and late == (READY_DEADLINE,)
    _, _, _, ok = evaluate_unit(load(required_ready_min=90), unit, edges())
    assert ok == ()


def test_voltage_tolerance_allows_small_delta():
    unit = g(voltage_v=405)
    assert evaluate_unit(load(voltage_v=400), unit, edges())[3] == (VOLTAGE_MISMATCH,)
    assert evaluate_unit(load(voltage_v=400), unit, edges(),
                         voltage_tolerance_v=5)[3] == ()


def test_uncovered_summary_dedupes_reasons():
    stock = GeneratorInventory([
        g("g1", voltage_v=480),
        g("g2", voltage_v=480),
    ])
    plan = plan_generators([load()], stock, edges())
    assert plan.committed is False
    uc = plan.uncovered[0]
    assert uc.load_id == "l1"
    assert uc.reasons == (VOLTAGE_MISMATCH,)  # one summary code, not two
    assert len(uc.rejections) == 2


def test_no_unit_means_all_reasons_reported():
    plan = plan_generators([load()], [], edges())
    assert plan.committed is False
    assert plan.assignments == ()
    assert plan.uncovered[0].reasons == ()
    assert plan.uncovered[0].rejections == ()


def test_duplicate_ids_rejected():
    import pytest
    stock = GeneratorInventory([g()])
    with pytest.raises(ValueError):
        GeneratorInventory([g(), g()])
    with pytest.raises(ValueError):
        plan_generators([load("x"), load("x")], stock, edges())


def test_inventory_clone_independent():
    stock = GeneratorInventory([g()])
    plan = plan_generators([load()], stock, edges(), )  # commits
    second = plan_generators([load()], stock, edges())
    assert second.committed is False  # already consumed by first plan
    assert second.uncovered[0].reasons == (ALREADY_OCCUPIED,)
