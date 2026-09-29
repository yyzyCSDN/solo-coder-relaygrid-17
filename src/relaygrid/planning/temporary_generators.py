from __future__ import annotations

from dataclasses import dataclass

from relaygrid.crew.routing import RoadEdge, shortest_minutes

REASON_VOLTAGE = "voltage_mismatch"
REASON_KW = "insufficient_kw"
REASON_KVAR = "insufficient_kvar"
REASON_CONNECTOR = "connector_mismatch"
REASON_STARTING = "insufficient_starting_current"
REASON_FUEL = "insufficient_fuel"
REASON_UNREACHABLE = "unreachable"
REASON_EXHAUSTED = "no_units_available"


@dataclass(frozen=True)
class MobileGenSet:
    """A mobile (temporary) generator set available for deployment."""

    gen_id: str
    node: str
    kv_rated: float
    kw_rated: float
    kvar_rated: float
    connector: str
    starting_current_ka: float
    fuel_l: float
    burn_lph: float

    @property
    def endurance_h(self) -> float:
        if self.burn_lph <= 0:
            return float("inf")
        return self.fuel_l / self.burn_lph


@dataclass(frozen=True)
class CriticalLoad:
    """A critical load that needs temporary generation."""

    load_id: str
    node: str
    kv_required: float
    kw: float
    kvar: float
    connector: str
    starting_current_ka: float
    endurance_h: float
    priority: int = 0


@dataclass(frozen=True)
class Deployment:
    gen_id: str
    load_id: str
    travel_min: float
    hookup_min: float
    eta_min: float
    fuel_endurance_h: float
    fuel_margin_h: float


@dataclass(frozen=True)
class Rejection:
    load_id: str
    reasons: tuple[str, ...]
    nearest_eta_min: float | None


@dataclass(frozen=True)
class DeploymentPlan:
    assignments: tuple[Deployment, ...]
    uncovered: tuple[str, ...]
    rejections: tuple[Rejection, ...]
    covered_kw: float
    uncovered_kw: float

    def generator_for(self, load_id: str) -> str | None:
        for d in self.assignments:
            if d.load_id == load_id:
                return d.gen_id
        return None


def _non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def unit_reasons(
    gen: MobileGenSet,
    load: CriticalLoad,
    voltage_tol_pct: float,
    fuel_reserve_h: float,
) -> list[str]:
    """Return the compatibility check codes a unit fails for a load."""
    reasons: list[str] = []
    tol = abs(load.kv_required) * voltage_tol_pct / 100.0
    if abs(gen.kv_rated - load.kv_required) > tol:
        reasons.append(REASON_VOLTAGE)
    if gen.kw_rated + 1e-9 < load.kw:
        reasons.append(REASON_KW)
    if gen.kvar_rated + 1e-9 < load.kvar:
        reasons.append(REASON_KVAR)
    if gen.connector != load.connector:
        reasons.append(REASON_CONNECTOR)
    if gen.starting_current_ka + 1e-9 < load.starting_current_ka:
        reasons.append(REASON_STARTING)
    if gen.endurance_h + 1e-9 < load.endurance_h + fuel_reserve_h:
        reasons.append(REASON_FUEL)
    return reasons


def plan_generator_deployment(
    gensets: list[MobileGenSet],
    loads: list[CriticalLoad],
    edges: list[RoadEdge],
    hookup_min: float = 60.0,
    voltage_tol_pct: float = 0.0,
    fuel_reserve_h: float = 0.0,
) -> DeploymentPlan:
    """Assign mobile generators to critical loads.

    Loads are served in priority order (highest first, then largest). Each
    generator is assigned at most once; failed matches never consume a unit
    and the input inventories are never mutated. Uncovered loads carry the
    aggregated reasons why no available unit could be dispatched.
    """
    _non_negative(hookup_min, "hookup_min")
    _non_negative(voltage_tol_pct, "voltage_tol_pct")
    _non_negative(fuel_reserve_h, "fuel_reserve_h")
    for g in gensets:
        _non_negative(g.kw_rated, "kw_rated")
        _non_negative(g.kvar_rated, "kvar_rated")
        _non_negative(g.starting_current_ka, "starting_current_ka")
        _non_negative(g.fuel_l, "fuel_l")
        _non_negative(g.burn_lph, "burn_lph")
    for load in loads:
        _non_negative(load.kw, "kw")
        _non_negative(load.kvar, "kvar")
        _non_negative(load.starting_current_ka, "starting_current_ka")
        _non_negative(load.endurance_h, "endurance_h")
    gen_ids = [g.gen_id for g in gensets]
    load_ids = [l.load_id for l in loads]
    if len(set(gen_ids)) != len(gen_ids):
        raise ValueError("generator ids must be unique")
    if len(set(load_ids)) != len(load_ids):
        raise ValueError("load ids must be unique")

    used: set[str] = set()
    assignments: list[Deployment] = []
    rejections: list[Rejection] = []

    ordered = sorted(loads, key=lambda l: (-l.priority, -l.kw, l.load_id))
    for load in ordered:
        best_key: tuple[float, float, str] | None = None
        best_gen: MobileGenSet | None = None
        best_travel = 0.0
        reasons_seen: set[str] = set()
        nearest_eta: float | None = None
        for gen in gensets:
            if gen.gen_id in used:
                continue
            reasons = unit_reasons(gen, load, voltage_tol_pct, fuel_reserve_h)
            travel = shortest_minutes(gen.node, load.node, edges)
            if travel == float("inf"):
                reasons.append(REASON_UNREACHABLE)
            else:
                eta = travel + hookup_min
                if nearest_eta is None or eta < nearest_eta:
                    nearest_eta = eta
            if reasons:
                reasons_seen.update(reasons)
                continue
            eta = travel + hookup_min
            key = (eta, gen.kw_rated - load.kw, gen.gen_id)
            if best_key is None or key < best_key:
                best_key = key
                best_gen = gen
                best_travel = travel
        if best_gen is None:
            if not any(g.gen_id not in used for g in gensets):
                reasons_seen.add(REASON_EXHAUSTED)
            rejections.append(
                Rejection(load.load_id, tuple(sorted(reasons_seen)), nearest_eta)
            )
            continue
        used.add(best_gen.gen_id)
        assignments.append(
            Deployment(
                best_gen.gen_id,
                load.load_id,
                best_travel,
                hookup_min,
                best_travel + hookup_min,
                best_gen.endurance_h,
                best_gen.endurance_h - load.endurance_h,
            )
        )

    covered = {d.load_id for d in assignments}
    uncovered = tuple(l.load_id for l in ordered if l.load_id not in covered)
    rejection_by_id = {r.load_id: r for r in rejections}
    ordered_rejections = tuple(rejection_by_id[i] for i in uncovered)
    covered_kw = sum(l.kw for l in loads if l.load_id in covered)
    uncovered_kw = sum(l.kw for l in loads if l.load_id not in covered)
    return DeploymentPlan(
        tuple(assignments),
        uncovered,
        ordered_rejections,
        covered_kw,
        uncovered_kw,
    )
