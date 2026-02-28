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

# Generate native C++ telemetry report from `plot on` (example output path)
# artifacts/telemetry_Rocket.svg
```

> Windows note: unsigned EXEs can be blocked by SmartScreen/Defender ("Access is denied").
> If needed: right-click EXE -> Properties -> Unblock, or run `Unblock-File` in PowerShell, or code-sign the build.
> `plot on` now uses native C++ animated SVG output (`artifacts/telemetry_<Body>.svg`) with no Python process launch.

## Test

```bash
cd build
ctest --output-on-failure

# Optional: run NASA-reference sanity checks for Earth-Moon and Mercury
./tools/validate_against_nasa.sh --strict
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

# Generate native C++ telemetry report from `plot on` (example output path)
# artifacts/telemetry_Rocket.svg

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

## Writing Gravity-Lang syntax (quick start)

Use this template when writing new `.gravity` scripts:

```gravity
# 1) Declare bodies
sphere Earth at [0,0,0][m] mass 5.972e24[kg] radius 6371[km] fixed
sphere Moon  at [384400,0,0][km] mass 7.348e22[kg] radius 1737[km]

# 2) Set optional properties
Moon.velocity = [0,1.022,0][km/s]

# 3) Add diagnostics/outputs (optional)
orbital_elements Moon around Earth
observe Moon.position to "artifacts/moon.csv" frequency 1
plot on body Moon

# 4) Run simulation (steps are END-START)
simulate orbit in 0..240 dt 3600[s] integrator rk45 {
    grav all
    print Moon.position
}
```

### Core syntax rules

- **Body declarations:** `sphere|probe|rocket Name at [x,y,z][unit] mass M[kg] radius R[unit] [velocity ...] [fixed]`.
- **Velocity assignment:** `Body.velocity = [vx,vy,vz][m/s|km/s]`.
- **Gravity rules:** use `grav all` for full N-body pull, or `A pull B, C` for explicit targets.
- **Simulation loop:** `simulate|orbit <label> in START..END dt VALUE[time_unit] integrator <name> { ... }`.
- **Orbital diagnostics:** always include a center body: `orbital_elements Body around Center`.

### Adding new code safely

1. Start from an existing example in `examples/` and rename it.
2. Run strict parser checks first:
   - `./build/gravity check your_script.gravity --strict`
3. Then run the simulation:
   - `./build/gravity run your_script.gravity`
4. If needed, export data for analysis:
   - `dump_all to "artifacts/dump.csv" frequency 1`
   - `observe Body.position to "artifacts/body_pos.csv" frequency 10`

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
- CSV export: `observe Body.position|velocity to "file" frequency N`, `dump_all to "file" frequency N`, DSL dashboard hook `plot on [body Name]`, and CLI `--dump-all[=file]` (output folders are auto-created if missing; animated native C++ SVG is generated automatically with `dump_all` unless `plot off` is explicitly set).
- Orbital diagnostics: `orbital_elements Body around Center`
- Confidence scores: adaptive runs print `confidence.score=...` based on timestep headroom and energy drift

## Notes

- This is an active port from the former Python implementation; unsupported DSL lines are still warned and skipped.
- Current focus is interpreter capability expansion and compatibility with existing `.gravity` scripts.
- CI prerelease tags: non-versioned CI publishes unique tags in the form `main-build-YYYYMMDD-HHMMSS-rRUN_NUMBER` to keep each release artifact set separate over time.


## Benchmark setup notes

- **Binary systems:** initialize both primaries from their mutual COM, not hardcoded guessed speeds. For separation `r`, use `omega = sqrt(G*(m1+m2)/r^3)`, then `v1 = omega*r*m2/(m1+m2)` and `v2 = omega*r*m1/(m1+m2)` in opposite tangential directions.
- **MOND galaxy tests:** for deep-MOND circular initialization, seed stars near `v_flat = (G*M*a0)^(1/4)` and then adjust for your chosen radius profile.
- **Rocket ascent tests:** treat gravity-turn and throttle schedules as trajectory design inputs; reaching orbit generally requires early pitch-over plus enough sustained horizontal burn time.

## Common DSL pitfalls (quick fixes)

- `orbital_elements` requires both body and center: `orbital_elements Moon around Earth`.
- `simulate/orbit in START..END` uses `(END - START)` as the step count, not elapsed seconds. Total simulated time is `steps * dt`.
- Extremely large step ranges can overflow internal limits; keep `END - START` within 32-bit integer bounds and use larger `dt` for long spans.
- `Body.fuel_mass = ...` contributes to total body mass (wet mass = dry/base mass + fuel mass).
- `grav all` now applies across all declared bodies regardless of where it appears in the script.
- `plot on` defaults to body `Rocket`; for other names, use `plot on body Name`.
