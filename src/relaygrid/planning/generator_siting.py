from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from relaygrid.crew.routing import RoadEdge, shortest_minutes

# Rejection reason codes (canonical order).
VOLTAGE_MISMATCH = "voltage_mismatch"
ACTIVE_POWER = "active_power"
APPARENT_POWER = "apparent_power"
STARTING_CURRENT = "starting_current"
OVERCURRENT = "overcurrent"
CONNECTOR_MISSING = "connector_missing"
FUEL_ENDURANCE = "fuel_endurance"
NO_ROUTE = "no_route"
READY_DEADLINE = "ready_deadline"
ALREADY_OCCUPIED = "already_occupied"

REASON_ORDER = (
    VOLTAGE_MISMATCH,
    ACTIVE_POWER,
    APPARENT_POWER,
    STARTING_CURRENT,
    OVERCURRENT,
    CONNECTOR_MISSING,
    FUEL_ENDURANCE,
    NO_ROUTE,
    READY_DEADLINE,
    ALREADY_OCCUPIED,
)


@dataclass(frozen=True)
class CriticalLoad:
    """A critical load that may need a temporary generator."""

    load_id: str
    node: str
    voltage_v: float
    active_kw: float
    reactive_kvar: float
    connector: str
    starting_a: float
    priority: int = 0
    required_hours: float = 0.0
    required_ready_min: float | None = None  # relative to dispatch_min
    phases: int = 3


@dataclass(frozen=True)
class GeneratorUnit:
    """A portable/trailer-mounted generator in the staging inventory."""

    unit_id: str
    location: str
    voltage_v: float
    rated_kw: float
    rated_kva: float
    sustained_a: float       # continuous output current rating
    inrush_a: float          # short-term current the set can deliver on motor start
    connectors: frozenset[str]
    tank_l: float
    burn_lph: float          # planned fuel consumption serving this kind of load
    hookup_min: float        # cable landing / 接入作业时间

    def endurance_h(self) -> float:
        return self.tank_l / self.burn_lph


@dataclass(frozen=True)
class Placement:
    load_id: str
    unit_id: str
    travel_min: float
    hookup_min: float
    ready_at_min: float
    operating_a: float
    endurance_h: float
    spare_kw: float


@dataclass(frozen=True)
class UnitRejection:
    unit_id: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class UncoveredLoad:
    load_id: str
    reasons: tuple[str, ...]               # de-duplicated, canonical order
    rejections: tuple[UnitRejection, ...]  # one entry per inventoried unit


@dataclass(frozen=True)
class DeploymentPlan:
    assignments: tuple[Placement, ...]
    uncovered: tuple[UncoveredLoad, ...]
    committed: bool

    @property
    def covered_load_ids(self) -> tuple[str, ...]:
        return tuple(p.load_id for p in self.assignments)


class GeneratorInventory:
    """Shareable stock of generators. Occupancy only changes on a committed plan."""

    def __init__(self, units: Iterable[GeneratorUnit]):
        self._units: dict[str, GeneratorUnit] = {}
        for u in units:
            _validate_unit(u)
            if u.unit_id in self._units:
                raise ValueError(f"duplicate generator unit id: {u.unit_id}")
            self._units[u.unit_id] = u
        self._available: set[str] = set(self._units)

    @property
    def units(self) -> tuple[GeneratorUnit, ...]:
        return tuple(self._units.values())

    @property
    def available_ids(self) -> frozenset[str]:
        return frozenset(self._available)

    def get(self, unit_id: str) -> GeneratorUnit:
        return self._units[unit_id]

    def occupy(self, unit_id: str) -> None:
        if unit_id not in self._units:
            raise KeyError(unit_id)
        self._available.discard(unit_id)

    def clone(self) -> "GeneratorInventory":
        other = GeneratorInventory(self._units.values())
        other._available = set(self._available)
        return other


def apparent_kva(load: CriticalLoad) -> float:
    return math.hypot(load.active_kw, load.reactive_kvar)


def operating_current_a(load: CriticalLoad) -> float:
    s_va = apparent_kva(load) * 1000.0
    if load.phases == 3:
        return s_va / (math.sqrt(3.0) * load.voltage_v)
    return s_va / load.voltage_v


def _validate_unit(u: GeneratorUnit) -> None:
    if not u.unit_id:
        raise ValueError("unit_id is required")
    for name, value in (
        ("voltage_v", u.voltage_v),
        ("rated_kw", u.rated_kw),
        ("rated_kva", u.rated_kva),
        ("sustained_a", u.sustained_a),
        ("inrush_a", u.inrush_a),
        ("hookup_min", u.hookup_min),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if u.voltage_v <= 0 or u.rated_kw <= 0 or u.rated_kva <= 0:
        raise ValueError("voltage_v, rated_kw and rated_kva must be positive")
    if u.tank_l < 0 or u.burn_lph <= 0:
        raise ValueError("tank_l must be >= 0 and burn_lph must be > 0")
    if not u.connectors:
        raise ValueError("at least one connector is required")


def _validate_load(load: CriticalLoad) -> None:
    if not load.load_id:
        raise ValueError("load_id is required")
    if load.phases not in (1, 3):
        raise ValueError("phases must be 1 or 3")
    for name, value in (
        ("voltage_v", load.voltage_v),
        ("active_kw", load.active_kw),
        ("reactive_kvar", load.reactive_kvar),
        ("starting_a", load.starting_a),
        ("required_hours", load.required_hours),
    ):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    if load.voltage_v <= 0:
        raise ValueError("voltage_v must be positive")


def evaluate_unit(
    load: CriticalLoad,
    unit: GeneratorUnit,
    edges: Iterable[RoadEdge] = (),
    *,
    dispatch_min: float = 0.0,
    voltage_tolerance_v: float = 0.0,
) -> tuple[float, float, float, tuple[str, ...]]:
    """Check every constraint for one load/unit pair.

    Returns ``(travel_min, ready_at_min, operating_a, reasons)``; an empty
    reasons tuple means the unit can serve the load.
    """
    edges = list(edges)
    reasons: list[str] = []
    if abs(unit.voltage_v - load.voltage_v) > voltage_tolerance_v:
        reasons.append(VOLTAGE_MISMATCH)
    if load.active_kw > unit.rated_kw:
        reasons.append(ACTIVE_POWER)
    if apparent_kva(load) > unit.rated_kva:
        reasons.append(APPARENT_POWER)
    if load.starting_a > unit.inrush_a:
        reasons.append(STARTING_CURRENT)
    iop = operating_current_a(load)
    if iop > unit.sustained_a:
        reasons.append(OVERCURRENT)
    if load.connector not in unit.connectors:
        reasons.append(CONNECTOR_MISSING)
    if load.required_hours > 0 and unit.endurance_h() < load.required_hours:
        reasons.append(FUEL_ENDURANCE)

    travel = shortest_minutes(unit.location, load.node, edges)
    ready_at = dispatch_min + travel + unit.hookup_min if travel != float("inf") else float("inf")
    if travel == float("inf"):
        reasons.append(NO_ROUTE)
    elif load.required_ready_min is not None and ready_at > load.required_ready_min:
        reasons.append(READY_DEADLINE)
    return travel, ready_at, iop, tuple(reasons)


def plan_generators(
    loads: Iterable[CriticalLoad],
    stock: GeneratorInventory | Iterable[GeneratorUnit],
    edges: Iterable[RoadEdge] = (),
    *,
    dispatch_min: float = 0.0,
    voltage_tolerance_v: float = 0.0,
) -> DeploymentPlan:
    """Match critical loads to non-reusable temporary generators.

    Loads are handled in priority order (highest priority first, then load_id).
    Each load gets the smallest feasible unit (rated kVA), breaking ties by the
    earliest ready time and then unit_id. Matching runs against a private
    working copy of the inventory; units reserved by earlier loads in the same
    batch cannot be reused. The shared inventory is mutated only when every
    load is covered, so a partial failure leaves stock untouched.
    """
    loads = list(loads)
    seen: set[str] = set()
    for load in loads:
        _validate_load(load)
        if load.load_id in seen:
            raise ValueError(f"duplicate load id: {load.load_id}")
        seen.add(load.load_id)

    inventory = stock if isinstance(stock, GeneratorInventory) else GeneratorInventory(stock)
    edges = list(edges)
    work = inventory.clone()

    placements: list[Placement] = []
    uncovered: list[UncoveredLoad] = []

    for load in sorted(loads, key=lambda x: (-x.priority, x.load_id)):
        feasible: list[tuple[float, float, str, GeneratorUnit, float]] = []
        rejections: list[UnitRejection] = []
        for unit in inventory.units:
            if unit.unit_id not in work.available_ids:
                rejections.append(UnitRejection(unit.unit_id, (ALREADY_OCCUPIED,)))
                continue
            travel, ready_at, iop, reasons = evaluate_unit(
                load,
                unit,
                edges,
                dispatch_min=dispatch_min,
                voltage_tolerance_v=voltage_tolerance_v,
            )
            if reasons:
                rejections.append(UnitRejection(unit.unit_id, reasons))
            else:
                feasible.append((unit.rated_kva, ready_at, unit.unit_id, unit, iop))

        if feasible:
            _, ready_at, _, unit, iop = min(feasible, key=lambda x: (x[0], x[1], x[2]))
            travel = ready_at - dispatch_min - unit.hookup_min
            work.occupy(unit.unit_id)
            placements.append(
                Placement(
                    load_id=load.load_id,
                    unit_id=unit.unit_id,
                    travel_min=travel,
                    hookup_min=unit.hookup_min,
                    ready_at_min=ready_at,
                    operating_a=iop,
                    endurance_h=unit.endurance_h(),
                    spare_kw=unit.rated_kw - load.active_kw,
                )
            )
        else:
            all_reasons = {r for rej in rejections for r in rej.reasons}
            summary = tuple(code for code in REASON_ORDER if code in all_reasons)
            rejections.sort(key=lambda r: r.unit_id)
            uncovered.append(UncoveredLoad(load.load_id, summary, tuple(rejections)))

    committed = not uncovered
    if committed:
        for placement in placements:
            inventory.occupy(placement.unit_id)
    return DeploymentPlan(tuple(placements), tuple(uncovered), committed)
