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
- **Scientific loop syntax**: supports both `orbit ... dt ... {}` and `simulate ... step ... {}`.
- **Observer pattern**: `observe Object.position|velocity to "file.csv" frequency N` streams step data to CSV.
- **Dimensional quantities**: `Quantity` operations support unit-aware dimensional consistency checks for algebraic expressions.
- **Extended primitives**: `sphere`, `cube`, `pointmass`, `probe`; optional `fixed` and `velocity` properties.

## Example (Scientific Loop)

```gravity
sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
sphere Moon at [384400[km],0,0] mass 7.348e22[kg] velocity [0,1[km],0]

simulate t in 0..24 step 3600[s] integrator leapfrog {
    Earth pull Moon
    print Moon.position
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
python gravity_lang_interpreter.py examples/earth_moon.gravity
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## Next Steps

1. Replace `PythonPhysicsBackend` with a real C++ kernel (pybind11) and true RK4 implementation.
2. Add Rust worker backend for parallel large-N body updates.
3. Add Go simulation control plane for remote orchestration.
4. Add C# editor/renderer with camera/lights/inspector.
5. Add Parquet/Arrow observers and plotting hooks (energy drift, phase diagrams).
