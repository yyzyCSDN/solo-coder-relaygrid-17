# RelayGrid

RelayGrid is a deterministic Python toolkit for distribution-grid operations.
It covers feeder topology, protection, voltage assessment, DER islanding,
switching safety, outage accounting, crew logistics, asset thermal limits,
telemetry quality, storage dispatch, and restoration sequencing.

The package is self-contained and uses only the Python standard library.

## Install and run

```bash
python -m pip install -e ".[test]"
relaygrid demo
python -m pytest -q
```

The demo builds a four-bus feeder with two open ties, reports its energized
state, safely closes the sectionalizing branch, and emits a JSON summary.

## Public API

The top-level package exports the feeder data model and `GridSimulator`:

```python
from relaygrid import Branch, Bus, FeederSnapshot, GridSimulator, SwitchState
```

Domain modules remain independently importable so applications can combine
topology, protection, telemetry, restoration, reliability, planning, and field
operations without depending on a web framework or database.

## Temporary generator siting

`relaygrid.planning.generator_siting` matches critical loads to non-reusable
portable generators. Each candidate is checked against load voltage (with an
optional tolerance), active power, apparent power (`sqrt(P²+Q²)`), motor
starting current vs. the set's inrush rating, steady operating current vs. its
sustained rating, available connector/pad, fuel endurance (tank/burn vs.
required hours), road reachability (`crew.routing.shortest_minutes`) and the
ready-time deadline (travel + hookup). Loads are served in priority order and
get the smallest feasible unit. Matching runs on a private inventory clone, so
a unit is never double-booked within a batch, and the shared inventory is
occupied only when every load is covered — partial failures leave stock
untouched and report each uncovered load with per-unit rejection reasons:

```python
from relaygrid.planning.generator_siting import (
    CriticalLoad, GeneratorInventory, GeneratorUnit, plan_generators,
)
```

## License

RelayGrid is released under the MIT License.
