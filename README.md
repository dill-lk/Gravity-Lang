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

# Run rocket testing demo
./build/gravity run examples/rocket_testing.gravity

# Run single-file full feature showcase
./build/gravity run examples/all_features_one.gravity

# Plot altitude/speed telemetry from dump_all CSV
python tools/telemetry_dashboard.py artifacts/rocket_dump.csv --body Rocket
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

# Run rocket testing demo
./build/gravity run examples/rocket_testing.gravity

# Run single-file full feature showcase
./build/gravity run examples/all_features_one.gravity

# Plot altitude/speed telemetry from dump_all CSV
python tools/telemetry_dashboard.py artifacts/rocket_dump.csv --body Rocket

# Dump all bodies each step to CSV
./build/gravity run examples/moon_orbit.gravity --dump-all=artifacts/dump_all.csv

# Resume from a saved checkpoint
./build/gravity run examples/moon_orbit.gravity --resume artifacts/checkpoint.json

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

- Object declarations: `sphere`, `probe`, and `rocket`
- Units: `m`, `km`, `s`, `min`, `hour`, `day`, `kg`, `m/s`, `km/s`
- `radius`, `mass`, `velocity`, and `fixed` parsing
- Velocity assignment: `Body.velocity = [...]`
- Simulation loops: `simulate` / `orbit`
- Integrators: `euler`, `verlet`, `leapfrog`, `rk4`, `yoshida4`, `rk45`
- Gravity rules: `grav all`, `A pull B, C`, and `step_physics(A,B)`
- Gravity tuning: `gravity_constant`, `gravity_model newtonian|mond|gr_correction`
- Performance scaling: optional multithreaded force accumulation via `threads N|auto`, `threading min_interactions N`, or `GRAVITY_THREADS` environment variable (with reused per-thread buffers to reduce allocator overhead and a reusable low-contention thread-pool scheduler)
- Runtime actions: `thrust`, `event step N thrust Body by [...]`, `radiation_pressure Body by [ax,ay,az][m/s2]`, `friction`, `collisions on|off|merge`, `monitor energy`, `monitor momentum`, `monitor angular_momentum`, `verbose on|off`, `save "checkpoint.json" frequency N`, `resume "checkpoint.json"`, `sensitivity Body mass P%`, `merge_heat F`, `print ...position|velocity`, `profile on|off`, plus rocketry fields (`Body.dry_mass`, `Body.fuel_mass`, `Body.burn_rate`, `Body.max_thrust`, `Body.isp_sea_level`, `Body.isp_vacuum`, `Body.drag_coefficient`, `Body.cross_section`, `Body.throttle`, `throttle Body to maintain velocity V[m/s]`, `gravity_turn Body start A[m|km] end B[m|km] final_pitch DEG`) and staging (`event step N detach Stage from Rocket`)
- CSV export: `observe Body.position|velocity to "file" frequency N`, `dump_all to "file" frequency N`, and CLI `--dump-all[=file]`
- Orbital diagnostics: `orbital_elements Body around Center`
- Confidence scores: adaptive runs print `confidence.score=...` based on timestep headroom and energy drift

## Notes

- This is an active port from the former Python implementation; unsupported DSL lines are still warned and skipped.
- Current focus is interpreter capability expansion and compatibility with existing `.gravity` scripts.
- CI prerelease tags: non-versioned CI publishes unique tags in the form `main-build-YYYYMMDD-HHMMSS-rRUN_NUMBER` to keep each release artifact set separate over time.
