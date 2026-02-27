# Gravity-Lang (C++ Interpreter Rebuild)

This repository is being rebuilt around a **native C++ interpreter** for Gravity-Lang.

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Run

```bash
./build/gravity run examples/moon_orbit.gravity
```

## Test

```bash
cd build
ctest --output-on-failure
```

## Binaries

- `gravity`: C++ interpreter/runtime.
- `gravityc`: C++ source emitter/compiler utility.

## Implemented runtime features (ported toward Python parity)

- Object declarations: `sphere` and `probe`
- Units: `m`, `km`, `s`, `min`, `hour`, `day`, `kg`, `m/s`, `km/s`
- `radius`, `mass`, `velocity`, and `fixed` parsing
- Velocity assignment: `Body.velocity = [...]`
- Simulation loops: `simulate` / `orbit`
- Integrators: `euler`, `verlet`, `leapfrog`, `rk4`
- Gravity rules: `grav all`, `A pull B, C`, and `step_physics(A,B)`
- Gravity tuning: `gravity_constant`, `gravity_model newtonian|mond|gr_correction`
- Runtime actions: `thrust`, `friction`, `collisions on`, `monitor energy`, `monitor momentum`, `monitor angular_momentum`, `print ...position|velocity`
- CSV export: `observe Body.position|velocity to "file" frequency N`
- Orbital diagnostics: `orbital_elements Body around Center`

## Notes

- This is an active port from the former Python implementation; unsupported DSL lines are still warned and skipped.
- Current focus is interpreter capability expansion and compatibility with existing `.gravity` scripts.
