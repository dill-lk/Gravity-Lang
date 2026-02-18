#!/usr/bin/env python3
"""
Demo of custom gravity laws in Gravity-Lang

This script demonstrates how to use custom gravity laws including:
1. Standard Newtonian gravity
2. Modified Newtonian Dynamics (MOND)
3. General Relativity corrections (Schwarzschild)
"""

import sys
sys.path.insert(0, '/home/runner/work/Gravity-Lang/Gravity-Lang')

from gravity_lang_interpreter import (
    GravityInterpreter,
    PythonPhysicsBackend,
    NumPyPhysicsBackend,
    newtonian_gravity,
    modified_newtonian_gravity,
    schwarzschild_correction,
)

# Example 1: Using Newtonian gravity (default)
print("=" * 60)
print("Example 1: Standard Newtonian Gravity")
print("=" * 60)

script1 = """
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed
sphere Moon at [384400,0,0][km] mass 7.348e22[kg] velocity [0,1.022,0][km/s]

simulate t in 0..5 step 3600[s] integrator verlet {
    Earth pull Moon
    print Moon.position
}
"""

interp1 = GravityInterpreter()
output1 = interp1.execute(script1)
print("\n".join(output1[:3]))  # Show first 3 outputs
print()

# Example 2: Using MOND gravity
print("=" * 60)
print("Example 2: Modified Newtonian Dynamics (MOND)")
print("=" * 60)

interp2 = GravityInterpreter(
    physics_backend=PythonPhysicsBackend(gravity_law=modified_newtonian_gravity)
)
output2 = interp2.execute(script1)
print("\n".join(output2[:3]))  # Show first 3 outputs
print()

# Example 3: Using General Relativity corrections
print("=" * 60)
print("Example 3: General Relativity (Schwarzschild)")
print("=" * 60)

interp3 = GravityInterpreter(
    physics_backend=PythonPhysicsBackend(gravity_law=schwarzschild_correction)
)
output3 = interp3.execute(script1)
print("\n".join(output3[:3]))  # Show first 3 outputs
print()

# Example 4: Using NumPy backend for performance
print("=" * 60)
print("Example 4: NumPy Backend (High Performance)")
print("=" * 60)

try:
    interp4 = GravityInterpreter(physics_backend=NumPyPhysicsBackend())
    output4 = interp4.execute(script1)
    print("\n".join(output4[:3]))  # Show first 3 outputs
    print("✓ NumPy backend is available and working!")
except RuntimeError as e:
    print(f"⚠ NumPy not available: {e}")
print()

# Example 5: Custom gravity law
print("=" * 60)
print("Example 5: Custom Gravity Law (Inverse-cube for demo)")
print("=" * 60)


def inverse_cube_gravity(mass: float, distance: float, **kwargs) -> float:
    """Example custom gravity law: a = G * M / r^3 (not physical, just demo)"""
    G = 6.67430e-11
    return G * mass / (distance ** 3)


interp5 = GravityInterpreter(
    physics_backend=PythonPhysicsBackend(gravity_law=inverse_cube_gravity)
)
output5 = interp5.execute(script1)
print("\n".join(output5[:3]))  # Show first 3 outputs
print()

print("=" * 60)
print("All examples completed successfully!")
print("=" * 60)
