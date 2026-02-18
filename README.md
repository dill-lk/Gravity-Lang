# Gravity-Lang

A scientific prototype of **Gravity 3D / Gravity Lang** for physics-driven simulation scripting.

## Hybrid Architecture Direction

| Language | Role |
|---|---|
| Python | Frontend language parser + orchestration layer |
| C++ | Core physics kernel (heavy math, collisions) |
| Rust | Memory-safe parallel compute modules |
| Go | Distributed/cloud simulation API |
| C# | GUI/editor and 3D visualization tooling |

The current repository focuses on the Python prototype, but the runtime is now shaped around a physics backend seam for migration toward C++/Rust kernels.

## Implemented v2 Scientific Concepts

- **Physics kernel integrators**: loop-level `integrator leapfrog|rk4` selection (default: `leapfrog`).
- **RK4 implementation**: `rk4` now runs an actual 4th-order Runge-Kutta timestep for position/velocity evolution.
- **Scientific loop syntax**: supports both `orbit ... dt ... {}` and `simulate ... step ... {}`.
- **3D runtime controls**: `grav all`, `friction <value>`, and `collisions on|off` are supported inside/outside loops.
- **Propulsion controls**: `thrust <Object> by [vx,vy,vz][m/s]` applies script-level delta-v for maneuvers.
- **System monitoring**: `monitor energy` records total system energy (kinetic + pairwise gravitational potential).
- **Explicit physics step**: `step_physics(Target, Source)` forces a pairwise 3D gravity update for that timestep.
- **Observer pattern**: `observe Object.position|velocity to "file.csv" frequency N` streams step data to CSV.
- **Dimensional quantities**: `Quantity` operations support unit-aware dimensional consistency checks for algebraic expressions.
- **Extended primitives**: `sphere`, `cube`, `pointmass`, `probe`; optional `fixed` and `velocity` properties.
- **Vector suffix units**: vectors can be written as `[x,y,z][m]`, `[x,y,z][km]`, `[x,y,z][m/s]`, `[x,y,z][km/s]`.

## Example (3D Orbital Setup)

```gravity
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed
sphere Moon at [384400,0,0][km] radius 1737[km] mass 7.348e22[kg]

Moon.velocity = [0,1.022,0][km/s]

orbit t in 0..24 dt 3600[s] integrator rk4 {
    step_physics(Moon, Earth)
    print Moon.position
}
```

The tangential velocity term is required for orbital motion; without it, the body falls radially. `step_physics(Moon, Earth)` performs an explicit 3D gravity update for that object pair each timestep.

## Example (Full Physics Block)

```gravity
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed
sphere Moon at [384400,0,0][km] radius 1737[km] mass 7.348e22[kg]
probe Satellite at [7000,0,0][km] mass 500[kg] velocity [0,7.5,0][km/s]

Moon.velocity = [0,1.022,0][km/s]

simulate t in 0..24 step 3600[s] integrator rk4 {
    grav all
    friction 0.000001
    collisions on
    print Moon.position
    print Satellite.position
    observe Moon.position to "artifacts/moon.csv" frequency 1
}
```

## Dimensional Quantity Example (Python API)

```python
interp = GravityInterpreter()
G = interp.parse_quantity("6.67430e-11[m^3 kg^-1 s^-2]")
M = interp.parse_quantity("5.972e24[kg]")
R = interp.parse_quantity("6371[km]")
acceleration = G.mul(M).div(R.pow(2))
```

`acceleration` yields dimensions `{L: 1, T: -2}` (m/s² equivalent).

## Run

```bash
python gravity_lang_interpreter.py run examples/earth_moon.gravity
```

Backward-compatible direct mode still works:

```bash
python gravity_lang_interpreter.py examples/earth_moon.gravity
```

## CLI (v1.0)

```bash
# validate script syntax/runtime references
python gravity_lang_interpreter.py check examples/earth_moon.gravity

# show version
python gravity_lang_interpreter.py --version
```

## Build Executable (v1.0)

Use the built-in CLI command to create a standalone executable via PyInstaller:

```bash
python -m pip install pyinstaller
python gravity_lang_interpreter.py build-exe --name gravity-lang-v1.0 --outdir dist --install-pyinstaller
```

This emits `dist/gravity-lang-v1.0` (or `dist/gravity-lang-v1.0.exe` on Windows). Use `--no-clean` if you want faster iterative local builds.

python gravity_lang_interpreter.py examples/earth_moon.gravity
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Next Steps

1. Replace `PythonPhysicsBackend` with a C++ kernel (pybind11) while keeping parser semantics stable.
2. Add optional NumPy/CuPy vectorized execution paths for large N-body systems.
3. Add Rust worker backend for parallel large-N body updates.
4. Add Go simulation control plane for remote orchestration.
5. Add C# editor/renderer with camera/lights/inspector.
6. Add Parquet/Arrow observers and plotting hooks (energy drift, phase diagrams).
1. Replace `PythonPhysicsBackend` with a real C++ kernel (pybind11) and true RK4 implementation.
2. Add Rust worker backend for parallel large-N body updates.
3. Add Go simulation control plane for remote orchestration.
4. Add C# editor/renderer with camera/lights/inspector.
5. Add Parquet/Arrow observers and plotting hooks (energy drift, phase diagrams).
