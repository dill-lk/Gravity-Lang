from __future__ import annotations

import csv
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Protocol, Tuple

G = 6.67430e-11
GRAVITY_LANG_VERSION = "1.0.0"

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

VECTOR_UNIT_SCALE = {
    "m": 1.0,
    "km": 1000.0,
    "m/s": 1.0,
    "km/s": 1000.0,
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
Integrator = Literal["leapfrog", "rk4", "verlet", "euler"]


def v_add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(v: Vec3, s: float) -> Vec3:
    return (v[0] * s, v[1] * s, v[2] * s)


def v_mag(v: Vec3) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def v_dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(v: Vec3) -> Vec3:
    m = v_mag(v)
    if m <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / m, v[1] / m, v[2] / m)


def v_distance(a: Vec3, b: Vec3) -> float:
    """Calculate distance between two points."""
    return v_mag(v_sub(a, b))


def v_angle(a: Vec3, b: Vec3) -> float:
    """Calculate angle between two vectors in radians."""
    mag_a = v_mag(a)
    mag_b = v_mag(b)
    if mag_a <= 1e-12 or mag_b <= 1e-12:
        return 0.0
    cos_angle = v_dot(a, b) / (mag_a * mag_b)
    # Clamp to avoid numerical errors
    cos_angle = max(-1.0, min(1.0, cos_angle))
    return math.acos(cos_angle)


@dataclass(frozen=True)
class OrbitalElements:
    """Orbital elements for a two-body system."""
    semi_major_axis: float  # meters
    eccentricity: float  # dimensionless
    inclination: float  # radians
    longitude_ascending_node: float  # radians (Ω)
    argument_periapsis: float  # radians (ω)
    true_anomaly: float  # radians (ν)
    
    @property
    def periapsis(self) -> float:
        """Distance at closest approach (meters)."""
        return self.semi_major_axis * (1 - self.eccentricity)
    
    @property
    def apoapsis(self) -> float:
        """Distance at farthest point (meters)."""
        return self.semi_major_axis * (1 + self.eccentricity)


def calculate_orbital_elements(position: Vec3, velocity: Vec3, central_mass: float) -> OrbitalElements:
    """
    Calculate Keplerian orbital elements from state vectors.
    
    Args:
        position: Position vector relative to central body (m)
        velocity: Velocity vector (m/s)
        central_mass: Mass of central body (kg)
    
    Returns:
        OrbitalElements for the orbit
    """
    mu = G * central_mass  # Standard gravitational parameter
    
    r = v_mag(position)
    v = v_mag(velocity)
    
    # Specific orbital energy
    epsilon = v * v / 2 - mu / r
    
    # Semi-major axis (negative for hyperbolic orbits)
    a = -mu / (2 * epsilon) if abs(epsilon) > 1e-12 else float('inf')
    
    # Angular momentum vector
    h_vec = v_cross(position, velocity)
    h = v_mag(h_vec)
    
    # Eccentricity vector
    rv_cross = v_cross(velocity, h_vec)
    e_vec = v_sub(v_scale(rv_cross, 1.0 / mu), v_norm(position))
    e = v_mag(e_vec)
    
    # Inclination
    i = math.acos(max(-1.0, min(1.0, h_vec[2] / h))) if h > 1e-12 else 0.0
    
    # Node vector (points to ascending node)
    k_vec = (0.0, 0.0, 1.0)
    n_vec = v_cross(k_vec, h_vec)
    n = v_mag(n_vec)
    
    # Longitude of ascending node
    if n > 1e-12:
        omega_lan = math.acos(max(-1.0, min(1.0, n_vec[0] / n)))
        if n_vec[1] < 0:
            omega_lan = 2 * math.pi - omega_lan
    else:
        omega_lan = 0.0
    
    # Argument of periapsis
    if n > 1e-12 and e > 1e-12:
        omega = math.acos(max(-1.0, min(1.0, v_dot(n_vec, e_vec) / (n * e))))
        if e_vec[2] < 0:
            omega = 2 * math.pi - omega
    else:
        omega = 0.0
    
    # True anomaly
    if e > 1e-12:
        nu = math.acos(max(-1.0, min(1.0, v_dot(e_vec, position) / (e * r))))
        if v_dot(position, velocity) < 0:
            nu = 2 * math.pi - nu
    else:
        nu = 0.0
    
    return OrbitalElements(
        semi_major_axis=a,
        eccentricity=e,
        inclination=i,
        longitude_ascending_node=omega_lan,
        argument_periapsis=omega,
        true_anomaly=nu,
    )


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
    def _accelerations_for_positions(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        positions: Dict[str, Vec3],
    ) -> Dict[str, Vec3]:
        accelerations: Dict[str, Vec3] = {name: (0.0, 0.0, 0.0) for name in objects}
        for source_name, target_name in pull_pairs:
            source = objects[source_name]
            displacement = v_sub(positions[source_name], positions[target_name])
            r = max(v_mag(displacement), 1e-9)
            acc_mag = G * source.mass / (r**2)
            acc = v_scale(v_norm(displacement), acc_mag)
            accelerations[target_name] = v_add(accelerations[target_name], acc)
        return accelerations

    def _accelerations(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]]) -> Dict[str, Vec3]:
        positions = {name: body.position for name, body in objects.items()}
        return self._accelerations_for_positions(objects, pull_pairs, positions)

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
        movable = [name for name, body in objects.items() if not body.fixed]
        base_pos = {name: body.position for name, body in objects.items()}
        base_vel = {name: body.velocity for name, body in objects.items()}

        def shifted_positions(k_pos: Dict[str, Vec3], scale: float) -> Dict[str, Vec3]:
            positions = dict(base_pos)
            for name in movable:
                positions[name] = v_add(base_pos[name], v_scale(k_pos[name], scale))
            return positions

        def shifted_velocities(k_vel: Dict[str, Vec3], scale: float) -> Dict[str, Vec3]:
            velocities = dict(base_vel)
            for name in movable:
                velocities[name] = v_add(base_vel[name], v_scale(k_vel[name], scale))
            return velocities

        acc1 = self._accelerations_for_positions(objects, pull_pairs, base_pos)
        k1_r = {name: base_vel[name] for name in movable}
        k1_v = {name: acc1[name] for name in movable}

        pos2 = shifted_positions(k1_r, dt * 0.5)
        vel2 = shifted_velocities(k1_v, dt * 0.5)
        acc2 = self._accelerations_for_positions(objects, pull_pairs, pos2)
        k2_r = {name: vel2[name] for name in movable}
        k2_v = {name: acc2[name] for name in movable}

        pos3 = shifted_positions(k2_r, dt * 0.5)
        vel3 = shifted_velocities(k2_v, dt * 0.5)
        acc3 = self._accelerations_for_positions(objects, pull_pairs, pos3)
        k3_r = {name: vel3[name] for name in movable}
        k3_v = {name: acc3[name] for name in movable}

        pos4 = shifted_positions(k3_r, dt)
        vel4 = shifted_velocities(k3_v, dt)
        acc4 = self._accelerations_for_positions(objects, pull_pairs, pos4)
        k4_r = {name: vel4[name] for name in movable}
        k4_v = {name: acc4[name] for name in movable}

        for name in movable:
            new_pos = v_add(
                base_pos[name],
                v_scale(
                    v_add(v_add(k1_r[name], v_scale(v_add(k2_r[name], k3_r[name]), 2.0)), k4_r[name]),
                    dt / 6.0,
                ),
            )
            new_vel = v_add(
                base_vel[name],
                v_scale(
                    v_add(v_add(k1_v[name], v_scale(v_add(k2_v[name], k3_v[name]), 2.0)), k4_v[name]),
                    dt / 6.0,
                ),
            )
            objects[name].position = new_pos
            objects[name].velocity = new_vel

    def _step_verlet(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        """Velocity Verlet integrator - symplectic, time-reversible, 2nd order accurate."""
        accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            
            # Update position: x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            body.position = v_add(
                body.position,
                v_add(
                    v_scale(body.velocity, dt),
                    v_scale(accelerations[name], 0.5 * dt * dt)
                )
            )
        
        # Calculate new accelerations at new positions
        new_accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            
            # Update velocity: v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
            body.velocity = v_add(
                body.velocity,
                v_scale(
                    v_add(accelerations[name], new_accelerations[name]),
                    0.5 * dt
                )
            )

    def _step_euler(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        """Simple Euler integrator - 1st order, less accurate but fast."""
        accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            
            # Update velocity: v(t+dt) = v(t) + a(t)*dt
            body.velocity = v_add(body.velocity, v_scale(accelerations[name], dt))
            
            # Update position: x(t+dt) = x(t) + v(t+dt)*dt
            body.position = v_add(body.position, v_scale(body.velocity, dt))

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
        if integrator == "verlet":
            self._step_verlet(objects, pull_pairs, dt)
            return
        if integrator == "euler":
            self._step_euler(objects, pull_pairs, dt)
            return
        raise ValueError(f"Unsupported integrator: {integrator}")


class GravityInterpreter:
    def __init__(self, physics_backend: PhysicsBackend | None = None) -> None:
        self.objects: Dict[str, Body] = {}
        self.pull_pairs: List[Tuple[str, str]] = []
        self.output: List[str] = []
        self.observers: List[Observer] = []
        self.global_friction = 0.0
        self.enable_collisions = True
        self.physics_backend = physics_backend or PythonPhysicsBackend()

    def _format_float(self, value: float) -> str:
        return f"{value:.6e}"

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
        token = token.strip()
        m = re.fullmatch(r"(\[[^\]]+\])(?:\[([a-zA-Z/]+)\])?", token)
        if not m:
            raise ValueError(f"Invalid vector token: {token}")

        vector_text = m.group(1)
        unit = m.group(2)
        inner = vector_text[1:-1]
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Vector must have 3 components: {token}")

        if unit:
            if unit not in VECTOR_UNIT_SCALE:
                raise ValueError(f"Unsupported vector unit: {unit}")
            if any("[" in part for part in parts):
                raise ValueError("Vector components must be unitless when using vector suffix units")
            scale = VECTOR_UNIT_SCALE[unit]
            return (float(parts[0]) * scale, float(parts[1]) * scale, float(parts[2]) * scale)

        return (self.parse_value(parts[0]), self.parse_value(parts[1]), self.parse_value(parts[2]))

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
        lines = []
        for raw_line in source.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith(("sphere ", "cube ", "pointmass ", "probe ")):
                self._parse_object(line)
                i += 1
            elif ".velocity" in line and "=" in line:
                self._parse_velocity_assignment(line)
                i += 1
            elif line == "grav all":
                self._add_gravity_all_pairs()
                i += 1
            elif line.startswith("friction "):
                self._parse_friction(line)
                i += 1
            elif line.startswith("collisions "):
                self._parse_collisions(line)
                i += 1
            elif line.startswith("thrust "):
                self._parse_thrust(line)
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
            elif line == "monitor energy":
                self._exec_monitor_energy()
                i += 1
            elif line.startswith("observe "):
                self._parse_observe(line)
                i += 1
            elif line.startswith("orbital_elements "):
                self._exec_orbital_elements(line)
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

        if trailing.startswith("["):
            vector_suffix, trailing = self._split_leading_vector(trailing)
            position_token = f"{position_token}{vector_suffix}"

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
            velocity_vector, vel_remaining = self._split_leading_vector(vel_tail)
            if vel_remaining.startswith("["):
                vel_unit, _ = self._split_leading_vector(vel_remaining)
                velocity_vector = f"{velocity_vector}{vel_unit}"
            velocity = self.parse_vector(velocity_vector)

        self.objects[name.strip()] = Body(
            name=name.strip(),
            shape=shape,
            position=position,
            velocity=velocity,
            radius=radius,
            mass=mass,
            fixed=fixed,
        )

    def _parse_velocity_assignment(self, line: str) -> None:
        m = re.fullmatch(r"(\w+)\.velocity\s*=\s*(.+)", line)
        if not m:
            raise ValueError(f"Invalid velocity assignment syntax: {line}")
        obj_name = m.group(1)
        body = self._require_object(obj_name, "velocity assignment")
        body.velocity = self.parse_vector(m.group(2).strip())

    def _parse_step_physics(self, line: str) -> Tuple[str, str]:
        m = re.fullmatch(r"step_physics\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", line)
        if not m:
            raise ValueError(f"Invalid step_physics statement: {line}")
        target_name = m.group(1)
        source_name = m.group(2)
        self._require_object(target_name, "step_physics statement")
        self._require_object(source_name, "step_physics statement")
        return (source_name, target_name)

    def _add_gravity_all_pairs(self) -> None:
        names = list(self.objects.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair_a = (names[i], names[j])
                pair_b = (names[j], names[i])
                if pair_a not in self.pull_pairs:
                    self.pull_pairs.append(pair_a)
                if pair_b not in self.pull_pairs:
                    self.pull_pairs.append(pair_b)

    def _parse_friction(self, line: str) -> None:
        _, value_token = line.split(" ", 1)
        value_token = value_token.strip()
        try:
            self.global_friction = float(value_token)
        except ValueError as exc:
            raise ValueError(f"Invalid friction value: {line}") from exc
        if self.global_friction < 0:
            raise ValueError("Friction must be non-negative")

    def _parse_collisions(self, line: str) -> None:
        _, value = line.split(" ", 1)
        value = value.strip().lower()
        if value == "on":
            self.enable_collisions = True
            return
        if value == "off":
            self.enable_collisions = False
            return
        raise ValueError(f"Invalid collisions setting: {line}")

    def _parse_thrust(self, line: str) -> None:
        m = re.fullmatch(r"thrust\s+(\w+)\s+by\s+(.+)", line)
        if not m:
            raise ValueError(f"Invalid thrust statement: {line}")
        obj_name = m.group(1)
        delta_v = self.parse_vector(m.group(2).strip())
        body = self._require_object(obj_name, "thrust statement")
        if body.fixed:
            return
        body.velocity = v_add(body.velocity, delta_v)

    def _exec_monitor_energy(self) -> None:
        self.output.append(f"system.energy={self._format_float(self._total_energy())}")

    def _total_energy(self) -> float:
        kinetic = 0.0
        for body in self.objects.values():
            speed = v_mag(body.velocity)
            kinetic += 0.5 * body.mass * speed * speed

        potential = 0.0
        undirected_pairs: set[Tuple[str, str]] = set()
        for source_name, target_name in self.pull_pairs:
            if source_name == target_name:
                continue
            pair = tuple(sorted((source_name, target_name)))
            undirected_pairs.add(pair)

        for a_name, b_name in undirected_pairs:
            a = self.objects[a_name]
            b = self.objects[b_name]
            r = max(v_mag(v_sub(a.position, b.position)), 1e-9)
            potential -= G * a.mass * b.mass / r
        return kinetic + potential

    def _apply_friction(self, dt: float) -> None:
        if self.global_friction <= 0:
            return
        damping = max(0.0, 1.0 - self.global_friction * dt)
        for body in self.objects.values():
            if body.fixed:
                continue
            body.velocity = v_scale(body.velocity, damping)

    def _resolve_collisions(self) -> None:
        if not self.enable_collisions:
            return
        names = list(self.objects.keys())
        restitution = 1.0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = self.objects[names[i]]
                b = self.objects[names[j]]
                if a.radius <= 0 or b.radius <= 0:
                    continue
                delta = v_sub(b.position, a.position)
                dist = v_mag(delta)
                min_dist = a.radius + b.radius
                if dist >= min_dist:
                    continue

                normal = v_norm(delta if dist > 1e-12 else (1.0, 0.0, 0.0))
                overlap = min_dist - dist

                if not a.fixed and not b.fixed:
                    a.position = v_add(a.position, v_scale(normal, -overlap * 0.5))
                    b.position = v_add(b.position, v_scale(normal, overlap * 0.5))
                elif not a.fixed:
                    a.position = v_add(a.position, v_scale(normal, -overlap))
                elif not b.fixed:
                    b.position = v_add(b.position, v_scale(normal, overlap))

                rel_vel = v_sub(b.velocity, a.velocity)
                vel_along_normal = v_dot(rel_vel, normal)
                if vel_along_normal > 0:
                    continue

                inv_mass_a = 0.0 if a.fixed else 1.0 / a.mass
                inv_mass_b = 0.0 if b.fixed else 1.0 / b.mass
                denom = inv_mass_a + inv_mass_b
                if denom <= 0:
                    continue
                impulse_mag = -(1.0 + restitution) * vel_along_normal / denom
                impulse = v_scale(normal, impulse_mag)
                if not a.fixed:
                    a.velocity = v_sub(a.velocity, v_scale(impulse, inv_mass_a))
                if not b.fixed:
                    b.velocity = v_add(b.velocity, v_scale(impulse, inv_mass_b))

    def _run_loop(self, lines: List[str], start: int) -> int:
        header = lines[start]
        m_orbit = re.fullmatch(
            r"orbit\s+\w+\s+in\s+(\d+)\.\.(\d+)\s+dt\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4|verlet|euler))?\s*\{",
            header,
        )
        m_sim = re.fullmatch(
            r"simulate\s+\w+\s+in\s+(\d+)\.\.(\d+)\s+step\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4|verlet|euler))?\s*\{",
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
        has_explicit_step = any(stmt.startswith("step_physics") for stmt in block)

        step_index = 0
        for _ in range(begin, end):
            step_pairs = list(self.pull_pairs)
            
            # Process configuration statements
            for stmt in block:
                if stmt == "grav all":
                    self._add_gravity_all_pairs()
                    step_pairs = list(self.pull_pairs)
                    continue
                if stmt.startswith("friction "):
                    self._parse_friction(stmt)
                    continue
                if stmt.startswith("collisions "):
                    self._parse_collisions(stmt)
                    continue
                if stmt.startswith("thrust "):
                    self._parse_thrust(stmt)
                    continue
                if " pull " in stmt:
                    a, _, b = stmt.partition(" pull ")
                    source_name, target_name = a.strip(), b.strip()
                    self._require_object(source_name, "pull statement")
                    self._require_object(target_name, "pull statement")
                    pair = (source_name, target_name)
                    if pair not in self.pull_pairs:
                        self.pull_pairs.append(pair)
                    if pair not in step_pairs:
                        step_pairs.append(pair)

            # Execute physics step
            if not has_explicit_step:
                self.physics_backend.step(self.objects, step_pairs, dt, integrator)
                self._apply_friction(dt)
                self._resolve_collisions()

            # Process step_physics and other statements
            for stmt in block:
                if stmt.startswith("step_physics"):
                    pair = self._parse_step_physics(stmt)
                    self.physics_backend.step(self.objects, [pair], dt, integrator)
                    self._apply_friction(dt)
                    self._resolve_collisions()
                    continue
                if stmt.endswith(".velocity") and "=" in stmt:
                    self._parse_velocity_assignment(stmt)
                    continue
                if stmt.startswith("print "):
                    self._exec_print(stmt)
                    continue
                if stmt == "monitor energy":
                    self._exec_monitor_energy()
                    continue

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

    def _exec_orbital_elements(self, line: str) -> None:
        """Calculate and print orbital elements for object orbiting central body."""
        # Format: orbital_elements Object around CentralBody
        m = re.fullmatch(r"orbital_elements\s+(\w+)\s+around\s+(\w+)", line)
        if not m:
            raise ValueError(f"Invalid orbital_elements statement: {line}. Use: orbital_elements Object around CentralBody")
        
        obj_name = m.group(1)
        central_name = m.group(2)
        
        obj = self._require_object(obj_name, "orbital_elements statement")
        central = self._require_object(central_name, "orbital_elements statement")
        
        # Calculate relative position and velocity
        rel_pos = v_sub(obj.position, central.position)
        rel_vel = v_sub(obj.velocity, central.velocity)
        
        # Calculate orbital elements
        elements = calculate_orbital_elements(rel_pos, rel_vel, central.mass)
        
        # Format output
        output_lines = [
            f"{obj_name} orbital elements around {central_name}:",
            f"  Semi-major axis: {elements.semi_major_axis/1000:.2f} km",
            f"  Eccentricity: {elements.eccentricity:.6f}",
            f"  Inclination: {math.degrees(elements.inclination):.2f}°",
            f"  Periapsis: {elements.periapsis/1000:.2f} km",
            f"  Apoapsis: {elements.apoapsis/1000:.2f} km",
        ]
        self.output.extend(output_lines)

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


def run_script_file(script_path: str) -> List[str]:
    script = Path(script_path).read_text(encoding="utf-8")
    interpreter = GravityInterpreter()
    return interpreter.execute(script)


def check_script_file(script_path: str) -> None:
    run_script_file(script_path)


def build_executable(name: str, outdir: str, auto_install: bool = False, clean: bool = True) -> Path:
    pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller and auto_install:
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
        pyinstaller = shutil.which("pyinstaller")
    if not pyinstaller:
        raise RuntimeError(
            "PyInstaller is not installed. Install it with: python -m pip install pyinstaller"
        )

    cmd = [
        pyinstaller,
        "--onefile",
        "--name",
        name,
        "--distpath",
        outdir,
        "gravity_lang_interpreter.py",
    ]
    if clean:
        cmd.insert(2, "--clean")
    subprocess.run(cmd, check=True)
    return Path(outdir) / (f"{name}.exe" if sys.platform.startswith("win") else name)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Gravity Lang CLI (v1.0)")
    parser.add_argument("--version", action="version", version=f"Gravity Lang {GRAVITY_LANG_VERSION}")

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a .gravity script")
    run_parser.add_argument("run_file", help="Path to a .gravity script")

    check_parser = sub.add_parser("check", help="Parse and validate a .gravity script")
    check_parser.add_argument("check_file", help="Path to a .gravity script")

    exe_parser = sub.add_parser("build-exe", help="Build a standalone v1.0 interpreter executable")
    exe_parser.add_argument("--name", default="gravity-lang-v1.0", help="Executable name")
    exe_parser.add_argument("--outdir", default="dist", help="Output directory")
    exe_parser.add_argument(
        "--install-pyinstaller",
        action="store_true",
        help="Automatically install pyinstaller if missing",
    )
    exe_parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Disable PyInstaller clean mode",
    )

    parser.add_argument("legacy_file", nargs="?", help="Backward-compatible: run a .gravity script directly")

    args = parser.parse_args()

    if args.command == "run":
        output = run_script_file(args.run_file)
        print("\n".join(output))
        return 0

    if args.command == "check":
        check_script_file(args.check_file)
        print(f"OK: {args.check_file}")
        return 0

    if args.command == "build-exe":
        output_path = build_executable(
            args.name,
            args.outdir,
            auto_install=args.install_pyinstaller,
            clean=not args.no_clean,
        )
        print(f"Built executable: {output_path}")
        return 0

    if args.legacy_file:
        output = run_script_file(args.legacy_file)
        print("\n".join(output))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Gravity Lang prototype interpreter")
    parser.add_argument("file", help="Path to a .gravity script")
    args = parser.parse_args()

    script = Path(args.file).read_text(encoding="utf-8")
    interpreter = GravityInterpreter()
    out = interpreter.execute(script)
    print("\n".join(out))
