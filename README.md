# Gravity-Lang

A first Python prototype of **Gravity Lang**, a physics-first language where objects live in 3D space and evolve over simulation timesteps.

## Implemented in this blueprint prototype

- Object declarations for `sphere` and `cube`
- Unit-aware numeric parsing (`[km]`, `[m]`, `[kg]`, `[s]`)
- One-way gravitational relation via `A pull B`
- Discrete simulation loops with `orbit ... dt ... { ... }`
- Printing `Object.position` values during simulation

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

## Next extensions

- Collision detection and response
- Symmetric N-body gravity management
- Reactive dependency graph
- Built-in rendering surface and camera language features
