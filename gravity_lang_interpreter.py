from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

G = 6.67430e-11

UNIT_SCALE = {
    "m": 1.0,
    "km": 1000.0,
    "s": 1.0,
    "kg": 1.0,
}

Vec3 = Tuple[float, float, float]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def v_mag(v: Vec3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def v_norm(v: Vec3) -> Vec3:
    m = v_mag(v)
    if m == 0:
        return (0.0, 0.0, 0.0)
    return (v[0] / m, v[1] / m, v[2] / m)


@dataclass
class Body:
    name: str
    shape: str
    position: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)
    mass: float = 1.0
    radius: float = 1.0
    properties: Dict[str, float] = field(default_factory=dict)


class PhysicsBackend(Protocol):
    """Interface expected from a physics engine backend (Python, C++, Rust, etc.)."""

    def step(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        ...


class PythonPhysicsBackend:
    """Reference physics backend in Python.

    Keeps the interpreter usable today while exposing a clean seam where C++/Rust backends
    can be connected later through ctypes/cffi/pybind11.
    """

    def step(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        accelerations: Dict[str, Vec3] = {name: (0.0, 0.0, 0.0) for name in objects}

        for source_name, target_name in pull_pairs:
            source = objects[source_name]
            target = objects[target_name]
            displacement = v_sub(source.position, target.position)
            r = max(v_mag(displacement), 1e-9)
            acc_mag = G * source.mass / (r**2)
            acc = v_scale(v_norm(displacement), acc_mag)
            accelerations[target_name] = v_add(accelerations[target_name], acc)

        for name, body in objects.items():
            new_velocity = v_add(body.velocity, v_scale(accelerations[name], dt))
            new_position = v_add(body.position, v_scale(new_velocity, dt))
            body.velocity = new_velocity
            body.position = new_position


class GravityInterpreter:
    def __init__(self, physics_backend: PhysicsBackend | None = None) -> None:
        self.objects: Dict[str, Body] = {}
        self.pull_pairs: List[Tuple[str, str]] = []
        self.output: List[str] = []
        self.physics_backend = physics_backend or PythonPhysicsBackend()

    def parse_value(self, token: str) -> float:
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?:\[([a-zA-Z]+)\])?", token)
        if not m:
            raise ValueError(f"Invalid numeric token: {token}")
        value = float(m.group(1))
        unit = m.group(2)
        if unit:
            if unit not in UNIT_SCALE:
                raise ValueError(f"Unsupported unit: {unit}")
            value *= UNIT_SCALE[unit]
        return value

    def parse_vector(self, token: str) -> Vec3:
        m = re.fullmatch(r"\[\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^\]]+)\s*\]", token)
        if not m:
            raise ValueError(f"Invalid vector token: {token}")
        return (self.parse_value(m.group(1)), self.parse_value(m.group(2)), self.parse_value(m.group(3)))

    def execute(self, source: str) -> List[str]:
        lines = [ln.strip() for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(("sphere ", "cube ")):
                self._parse_object(line)
                i += 1
            elif " pull " in line:
                a, _, b = line.partition(" pull ")
                self.pull_pairs.append((a.strip(), b.strip()))
                i += 1
            elif line.startswith("orbit "):
                i = self._run_orbit(lines, i)
            elif line.startswith("print "):
                self._exec_print(line)
                i += 1
            else:
                raise ValueError(f"Unsupported statement: {line}")
        return self.output

    def _parse_object(self, line: str) -> None:
        shape, rest = line.split(" ", 1)
        name, rest = rest.split(" at ", 1)
        pos_token, rest = rest.split(" radius ", 1)
        radius_token, rest = rest.split(" mass ", 1)

        self.objects[name.strip()] = Body(
            name=name.strip(),
            shape=shape,
            position=self.parse_vector(pos_token.strip()),
            radius=self.parse_value(radius_token.strip()),
            mass=self.parse_value(rest.strip()),
        )

    def _run_orbit(self, lines: List[str], start: int) -> int:
        header = lines[start]
        m = re.fullmatch(r"orbit\s+\w+\s+in\s+(\d+)\.\.(\d+)\s+dt\s+([^\s]+)\s*\{", header)
        if not m:
            raise ValueError(f"Invalid orbit statement: {header}")
        begin = int(m.group(1))
        end = int(m.group(2))
        dt = self.parse_value(m.group(3))

        block: List[str] = []
        i = start + 1
        while i < len(lines) and lines[i] != "}":
            block.append(lines[i])
            i += 1
        if i == len(lines):
            raise ValueError("orbit block missing closing brace")

        for _ in range(begin, end):
            self.physics_backend.step(self.objects, self.pull_pairs, dt)
            for stmt in block:
                if stmt.startswith("print "):
                    self._exec_print(stmt)
        return i + 1

    def _exec_print(self, line: str) -> None:
        expr = line.replace("print", "", 1).strip()
        if expr.endswith(".position"):
            obj_name = expr[: -len(".position")]
            body = self.objects[obj_name]
            self.output.append(f"{obj_name}.position={body.position}")
            return
        raise ValueError(f"Unsupported print expression: {expr}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Gravity Lang prototype interpreter")
    parser.add_argument("file", help="Path to a .gravity script")
    args = parser.parse_args()

    script = Path(args.file).read_text()
    interpreter = GravityInterpreter()
    out = interpreter.execute(script)
    print("\n".join(out))
