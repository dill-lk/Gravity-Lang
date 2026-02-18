# Gravity-Lang

A first prototype of **Gravity Lang**, a physics-first language where objects live in 3D space and evolve over simulation timesteps.

## Updated Blueprint Direction (Hybrid Multi-language)

| Language | Role in Gravity Lang |
|---|---|
| C++ | Core physics engine, heavy math, collision detection |
| Rust | Memory-safe parallel computations and optional physics modules |
| Go | Networking, cloud simulations, distributed orchestration |
| C# | GUI/editor, 3D visualization tooling |
| Python | Frontend scripting/parsing layer for fast iteration |

### Architecture sketch

```text
+------------------+       +----------------+       +----------------+
|  Python Frontend  | <---> |   C++ Physics  | <---> |  Rust Engine   |
|  (Gravity Script) |       |   Engine Core  |       | Parallel Tasks |
+------------------+       +----------------+       +----------------+
          |
          v
       +-------+
       | Go API |
       | Server |
       +-------+
          |
          v
       +-------+
       | C# GUI |
       | Editor |
       +-------+
```

This repository currently implements the **Python Frontend prototype**, with a backend interface seam that can be replaced by C++/Rust FFI.

## Implemented now (Python prototype)

- Object declarations for `sphere` and `cube`
- Unit-aware numeric parsing (`[km]`, `[m]`, `[kg]`, `[s]`)
- One-way gravitational relation via `A pull B`
- Discrete simulation loops with `orbit ... dt ... { ... }`
- Printing `Object.position`
- A pluggable `PhysicsBackend` protocol + `PythonPhysicsBackend` reference implementation

## Example script

```gravity
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg]
sphere Moon at [384400[km],0,0] radius 1737[km] mass 7.348e22[kg]

Earth pull Moon

orbit t in 0..24 dt 3600[s] {
    print Moon.position
}
```

See: `examples/earth_moon.gravity`.

## Run

```bash
python gravity_lang_interpreter.py examples/earth_moon.gravity
```

## Test

```bash
python -m unittest discover -s tests -v
```

## Next steps

1. Add C++ physics backend and Python bindings (pybind11/cffi).
2. Add Rust worker module for parallel particle/field updates.
3. Add Go simulation API for remote execution and orchestration.
4. Add C# editor/viewport with object inspector and timeline controls.
5. Extend language syntax (collisions, rotations, constraints, material models).
