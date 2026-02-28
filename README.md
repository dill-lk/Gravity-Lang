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

## CLI quick reference

```bash
# Run a simulation
./build/gravity run examples/moon_orbit.gravity

# Validate/parse script without running physics
./build/gravity check examples/moon_orbit.gravity --strict

# Show supported runtime features from the binary
./build/gravity list-features

# View CLI banner/help (includes ENGINE v3.0 ASCII banner)
./build/gravity --help
./build/gravityc --help

# Emit C++ from a Gravity script
./build/gravityc examples/moon_orbit.gravity --emit moon.cpp --strict
```

## Implemented runtime features (ported toward Python parity)

- Object declarations: `sphere` and `probe`
- Units: `m`, `km`, `s`, `min`, `hour`, `day`, `kg`, `m/s`, `km/s`
- `radius`, `mass`, `velocity`, and `fixed` parsing
- Velocity assignment: `Body.velocity = [...]`
- Simulation loops: `simulate` / `orbit`
- Integrators: `euler`, `verlet`, `leapfrog`, `rk4`
- Gravity rules: `grav all`, `A pull B, C`, and `step_physics(A,B)`
- Gravity tuning: `gravity_constant`, `gravity_model newtonian|mond|gr_correction`
- Performance scaling: optional multithreaded force accumulation via `threads N|auto`, `threading min_interactions N`, or `GRAVITY_THREADS` environment variable (with reused per-thread buffers to reduce allocator overhead)
- Runtime actions: `thrust`, `friction`, `collisions on`, `monitor energy`, `monitor momentum`, `monitor angular_momentum`, `print ...position|velocity`, `profile on|off`
- CSV export: `observe Body.position|velocity to "file" frequency N`
- Orbital diagnostics: `orbital_elements Body around Center`

## Notes

- This is an active port from the former Python implementation; unsupported DSL lines are still warned and skipped.
- Current focus is interpreter capability expansion and compatibility with existing `.gravity` scripts.
- CI prerelease tags: non-versioned CI publishes unique tags in the form `main-build-YYYYMMDD-HHMMSS-rRUN_NUMBER` to keep each release artifact set separate over time.
