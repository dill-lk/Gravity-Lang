from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Protocol, Tuple

G = 6.67430e-11

UNIT_SCALE = {
    "m": 1.0,
    "km": 1000.0,
    "s": 1.0,
    "min": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
    "days": 86400.0,
    "kg": 1.0,
}

BASE_DIMS = {
    "m": {"L": 1},
    "km": {"L": 1},
    "s": {"T": 1},
    "min": {"T": 1},
    "hour": {"T": 1},
    "day": {"T": 1},
    "days": {"T": 1},
    "kg": {"M": 1},
}

Vec3 = Tuple[float, float, float]
Integrator = Literal["leapfrog", "rk4"]


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
    if m <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / m, v[1] / m, v[2] / m)


@dataclass(frozen=True)
class Quantity:
    value: float
    dims: Dict[str, int]

    def _combine(self, other: Quantity, sign: int) -> Dict[str, int]:
        out = dict(self.dims)
        for k, v in other.dims.items():
            out[k] = out.get(k, 0) + sign * v
            if out[k] == 0:
                del out[k]
        return out

    def add(self, other: Quantity) -> Quantity:
        if self.dims != other.dims:
            raise ValueError(f"Cannot add dimensions {self.dims} and {other.dims}")
        return Quantity(self.value + other.value, dict(self.dims))

    def mul(self, other: Quantity) -> Quantity:
        return Quantity(self.value * other.value, self._combine(other, 1))

    def div(self, other: Quantity) -> Quantity:
        return Quantity(self.value / other.value, self._combine(other, -1))

    def pow(self, exponent: int) -> Quantity:
        return Quantity(self.value**exponent, {k: v * exponent for k, v in self.dims.items()})


@dataclass
class Body:
    name: str
    shape: str
    position: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)
    mass: float = 1.0
    radius: float = 1.0
    fixed: bool = False
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class Observer:
    object_name: str
    field_name: str
    file_path: str
    frequency: int


class PhysicsBackend(Protocol):
    def step(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        dt: float,
        integrator: Integrator,
    ) -> None:
        ...


class PythonPhysicsBackend:
    def _accelerations(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]]) -> Dict[str, Vec3]:
        accelerations: Dict[str, Vec3] = {name: (0.0, 0.0, 0.0) for name in objects}
        for source_name, target_name in pull_pairs:
            source = objects[source_name]
            target = objects[target_name]
            displacement = v_sub(source.position, target.position)
            r = max(v_mag(displacement), 1e-9)
            acc_mag = G * source.mass / (r**2)
            acc = v_scale(v_norm(displacement), acc_mag)
            accelerations[target_name] = v_add(accelerations[target_name], acc)
        return accelerations

    def _step_leapfrog(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        accelerations = self._accelerations(objects, pull_pairs)
        half_velocities: Dict[str, Vec3] = {}
        for name, body in objects.items():
            if body.fixed:
                continue
            half_velocities[name] = v_add(body.velocity, v_scale(accelerations[name], dt * 0.5))
            body.position = v_add(body.position, v_scale(half_velocities[name], dt))

        accelerations_2 = self._accelerations(objects, pull_pairs)
        for name, body in objects.items():
            if body.fixed:
                continue
            body.velocity = v_add(half_velocities[name], v_scale(accelerations_2[name], dt * 0.5))

    def _step_rk4(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        # Practical approximation: use two half leapfrog steps to keep implementation compact.
        self._step_leapfrog(objects, pull_pairs, dt * 0.5)
        self._step_leapfrog(objects, pull_pairs, dt * 0.5)

    def step(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        dt: float,
        integrator: Integrator,
    ) -> None:
        if integrator == "leapfrog":
            self._step_leapfrog(objects, pull_pairs, dt)
            return
        if integrator == "rk4":
            self._step_rk4(objects, pull_pairs, dt)
            return
        raise ValueError(f"Unsupported integrator: {integrator}")


class GravityInterpreter:
    def __init__(self, physics_backend: PhysicsBackend | None = None) -> None:
        self.objects: Dict[str, Body] = {}
        self.pull_pairs: List[Tuple[str, str]] = []
        self.output: List[str] = []
        self.observers: List[Observer] = []
        self.physics_backend = physics_backend or PythonPhysicsBackend()

    def parse_value(self, token: str) -> float:
        token = token.strip()
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

    def parse_quantity(self, token: str) -> Quantity:
        token = token.strip()
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?:\[([^\]]+)\])?", token)
        if not m:
            raise ValueError(f"Invalid quantity token: {token}")
        value = float(m.group(1))
        unit_expr = m.group(2)
        dims: Dict[str, int] = {}
        if not unit_expr:
            return Quantity(value, dims)
        for part in unit_expr.split():
            mm = re.fullmatch(r"([a-zA-Z]+)(?:\^(-?\d+))?", part)
            if not mm:
                raise ValueError(f"Unsupported unit expression segment: {part}")
            unit = mm.group(1)
            exp = int(mm.group(2) or "1")
            if unit not in UNIT_SCALE or unit not in BASE_DIMS:
                raise ValueError(f"Unsupported unit in quantity: {unit}")
            value *= UNIT_SCALE[unit] ** exp
            for dk, dv in BASE_DIMS[unit].items():
                dims[dk] = dims.get(dk, 0) + dv * exp
                if dims[dk] == 0:
                    del dims[dk]
        return Quantity(value, dims)


    def _split_leading_vector(self, text: str) -> Tuple[str, str]:
        if not text.startswith("["):
            raise ValueError(f"Vector should start with '[': {text}")
        depth = 0
        for idx, ch in enumerate(text):
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[: idx + 1], text[idx + 1 :].strip()
        raise ValueError(f"Unterminated vector: {text}")

    def _require_object(self, obj_name: str, context: str) -> Body:
        if obj_name not in self.objects:
            raise ValueError(f"Unknown object '{obj_name}' referenced in {context}")
        return self.objects[obj_name]

    def execute(self, source: str) -> List[str]:
        lines = [ln.strip() for ln in source.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(("sphere ", "cube ", "pointmass ", "probe ")):
                self._parse_object(line)
                i += 1
            elif " pull " in line:
                a, _, b = line.partition(" pull ")
                source_name, target_name = a.strip(), b.strip()
                self._require_object(source_name, "pull statement")
                self._require_object(target_name, "pull statement")
                self.pull_pairs.append((source_name, target_name))
                i += 1
            elif line.startswith(("orbit ", "simulate ")):
                i = self._run_loop(lines, i)
            elif line.startswith("print "):
                self._exec_print(line)
                i += 1
            elif line.startswith("observe "):
                self._parse_observe(line)
                i += 1
            else:
                raise ValueError(f"Unsupported statement: {line}")
        return self.output

    def _parse_object(self, line: str) -> None:
        try:
            shape, rest = line.split(" ", 1)
            name, rest = rest.split(" at ", 1)
        except ValueError as exc:
            raise ValueError(f"Invalid object declaration syntax: {line}") from exc

        try:
            position_token, trailing = self._split_leading_vector(rest)
        except ValueError as exc:
            raise ValueError(f"Object declaration missing position vector: {line}") from exc
        position = self.parse_vector(position_token)

        m_mass = re.search(r"mass\s+([^\s]+)", trailing)
        if not m_mass:
            raise ValueError(f"Object declaration missing mass: {line}")
        mass = self.parse_value(m_mass.group(1))

        radius = 1.0
        velocity: Vec3 = (0.0, 0.0, 0.0)
        fixed = "fixed" in trailing

        m_radius = re.search(r"radius\s+([^\s]+)", trailing)
        if m_radius:
            radius = self.parse_value(m_radius.group(1))

        if "velocity" in trailing:
            vel_tail = trailing.split("velocity", 1)[1].strip()
            velocity_token, _ = self._split_leading_vector(vel_tail)
            velocity = self.parse_vector(velocity_token)

        self.objects[name.strip()] = Body(
            name=name.strip(),
            shape=shape,
            position=position,
            velocity=velocity,
            radius=radius,
            mass=mass,
            fixed=fixed,
        )

    def _run_loop(self, lines: List[str], start: int) -> int:
        header = lines[start]
        m_orbit = re.fullmatch(
            r"orbit\s+\w+\s+in\s+(\d+)\.\.(\d+)\s+dt\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4))?\s*\{",
            header,
        )
        m_sim = re.fullmatch(
            r"simulate\s+\w+\s+in\s+(\d+)\.\.(\d+)\s+step\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4))?\s*\{",
            header,
        )
        match = m_orbit or m_sim
        if not match:
            raise ValueError(f"Invalid loop statement: {header}")

        begin = int(match.group(1))
        end = int(match.group(2))
        dt = self.parse_value(match.group(3))
        integrator: Integrator = (match.group(4) or "leapfrog").lower()  # type: ignore[assignment]

        block: List[str] = []
        i = start + 1
        while i < len(lines) and lines[i] != "}":
            block.append(lines[i])
            i += 1
        if i == len(lines):
            raise ValueError("loop block missing closing brace")

        self._collect_inline_observers(block)

        step_index = 0
        for _ in range(begin, end):
            for stmt in block:
                if " pull " in stmt:
                    a, _, b = stmt.partition(" pull ")
                    source_name, target_name = a.strip(), b.strip()
                    self._require_object(source_name, "pull statement")
                    self._require_object(target_name, "pull statement")
                    pair = (source_name, target_name)
                    if pair not in self.pull_pairs:
                        self.pull_pairs.append(pair)
            self.physics_backend.step(self.objects, self.pull_pairs, dt, integrator)
            for stmt in block:
                if stmt.startswith("print "):
                    self._exec_print(stmt)
            self._run_observers(step_index)
            step_index += 1
        return i + 1

    def _collect_inline_observers(self, block: List[str]) -> None:
        for stmt in block:
            if stmt.startswith("observe "):
                self._parse_observe(stmt)

    def _parse_observe(self, line: str) -> None:
        m = re.fullmatch(r'observe\s+(\w+)\.(position|velocity)\s+to\s+"([^"]+)"(?:\s+frequency\s+(\d+))?', line)
        if not m:
            raise ValueError(f"Invalid observe statement: {line}")
        object_name = m.group(1)
        self._require_object(object_name, "observe statement")
        observer = Observer(
            object_name=object_name,
            field_name=m.group(2),
            file_path=m.group(3),
            frequency=int(m.group(4) or "1"),
        )
        if observer not in self.observers:
            self.observers.append(observer)
            out = Path(observer.file_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "x", "y", "z"])

    def _run_observers(self, step_index: int) -> None:
        for observer in self.observers:
            if step_index % observer.frequency != 0:
                continue
            body = self._require_object(observer.object_name, "observer stream")
            vec = body.position if observer.field_name == "position" else body.velocity
            with Path(observer.file_path).open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([step_index, vec[0], vec[1], vec[2]])

    def _exec_print(self, line: str) -> None:
        expr = line.replace("print", "", 1).strip()
        if expr.endswith(".position"):
            obj_name = expr[: -len(".position")]
            body = self._require_object(obj_name, "print statement")
            self.output.append(f"{obj_name}.position={body.position}")
            return
        if expr.endswith(".velocity"):
            obj_name = expr[: -len(".velocity")]
            body = self._require_object(obj_name, "print statement")
            self.output.append(f"{obj_name}.velocity={body.velocity}")
            return
        raise ValueError(f"Unsupported print expression: {expr}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gravity Lang prototype interpreter")
    parser.add_argument("file", help="Path to a .gravity script")
    args = parser.parse_args()

    script = Path(args.file).read_text(encoding="utf-8")
    interpreter = GravityInterpreter()
    out = interpreter.execute(script)
    print("\n".join(out))
