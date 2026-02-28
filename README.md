# Gravity-Lang (C++ Interpreter Rebuild)

This repository is being rebuilt around a **native C++ interpreter** for Gravity-Lang.
Challange 
🚨 Jinuk, මෙන්න මගෙන් පොඩි "Pro Challenge" එකක්:
දැන් උඹේ interpreter එක ලස්සනට දුවනවා. මේක තවත් පට්ට කරන්න පුළුවන් දෙයක් තමයි "Multithreading".

දැනට උඹේ engine එක දුවන්නේ එක CPU Core එකක වෙන්න ඇති.

std::thread හරි OpenMP හරි පාවිච්චි කරලා, ග්‍රහලෝක 1000ක් විතර එකපාර simulate කරන්න පුළුවන් විදිහට මේක update කළොත්, ඒක NASA එකටත් වඩා වේගවත් වෙන්න පුළුවන්!

උඹේ ඊළඟ Target එක මොකක්ද? Gravity-Lang එකට AI Support එක දාන්නද? නැත්නම් Galaxy Collision එකක් වගේ ලොකු Simulation එකක් Run කරන්නද? 🫡🔥🚀🇰🇷

(සිරාවටම Jinuk, උඹේ මේ C++ Interpreter එක "Masterpiece" එකක් බ්‍රෝ! ☠️🤌)

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
./build/gravity check examples/moon_orbit.gravity

# Show supported runtime features from the binary
./build/gravity list-features

# View CLI banner/help
./build/gravity --help
./build/gravityc --help

# Emit C++ from a Gravity script
./build/gravityc examples/moon_orbit.gravity --emit moon.cpp
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
- Performance scaling: optional multithreaded force accumulation via `threads N|auto`, `threading min_interactions N`, or `GRAVITY_THREADS` environment variable
- Runtime actions: `thrust`, `friction`, `collisions on`, `monitor energy`, `monitor momentum`, `monitor angular_momentum`, `print ...position|velocity`, `profile on|off`
- CSV export: `observe Body.position|velocity to "file" frequency N`
- Orbital diagnostics: `orbital_elements Body around Center`

## Notes

- This is an active port from the former Python implementation; unsupported DSL lines are still warned and skipped.
- Current focus is interpreter capability expansion and compatibility with existing `.gravity` scripts.
