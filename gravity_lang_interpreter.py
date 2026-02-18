from __future__ import annotations

import csv
import ctypes
import os
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Protocol, Tuple

# Optional NumPy support for performance
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore
    HAS_NUMPY = False

# Optional matplotlib support for 3D visualization
try:
    import matplotlib
    if not os.environ.get("DISPLAY") and not sys.platform.startswith("win"):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None  # type: ignore
    HAS_MATPLOTLIB = False


# ============================================================================
# Custom Exception Classes for Better Error Handling
# ============================================================================

class GravityLangError(Exception):
    """Base exception for all Gravity-Lang errors."""
    def __init__(self, message: str, suggestion: str = "", line_number: int | None = None):
        self.message = message
        self.suggestion = suggestion
        self.line_number = line_number
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with colors and suggestions."""
        parts = []
        if self.line_number is not None:
            parts.append(f"❌ Error on line {self.line_number}")
        else:
            parts.append("❌ Error")
        parts.append(f"\n  {self.message}")
        if self.suggestion:
            parts.append(f"\n\n💡 Suggestion: {self.suggestion}")
        return "".join(parts)


class ParseError(GravityLangError):
    """Exception raised for syntax/parsing errors."""
    pass


class SimulationError(GravityLangError):
    """Exception raised during simulation execution."""
    pass


class UnitError(GravityLangError):
    """Exception raised for unit-related errors."""
    pass


class ObjectError(GravityLangError):
    """Exception raised for object-related errors (missing, duplicate, etc.)."""
    pass


class ValidationError(GravityLangError):
    """Exception raised for validation errors (negative mass, etc.)."""
    pass


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
BackendName = Literal["auto", "python", "numpy", "cpp", "julia_diffeq"]


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
    color: str = ""  # Optional color for visualization
    properties: Dict[str, float] = field(default_factory=dict)


@dataclass
class Observer:
    object_name: str
    field_name: str
    file_path: str
    frequency: int


class Visualizer3D:
    """3D visualization for gravitational simulations using matplotlib.
    
    Color format: Accepts matplotlib-compatible color names (e.g., 'blue', 'red')
    or hex codes (e.g., '#0000FF'). See matplotlib.colors for valid names.
    """
    
    # Default color palette for objects without explicit colors
    DEFAULT_COLOR_PALETTE = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    
    # Sphere size constants for visualization
    MIN_SPHERE_SIZE = 20
    MAX_SPHERE_SIZE = 200
    BASE_SPHERE_SIZE = 50
    SIZE_SCALE_FACTOR = 20
    
    def __init__(self, title: str = "Gravity Simulation", output_file: str = "gravity_simulation_3d.png", 
                 update_interval: int = 1):
        """Initialize 3D visualizer.
        
        Args:
            title: Title for the visualization window
            output_file: Filename to save the final visualization
            update_interval: Render every N simulation steps (default: 1)
        """
        if not HAS_MATPLOTLIB:
            raise RuntimeError(
                "matplotlib is required for 3D visualization. "
                "Install it with: pip install matplotlib"
            )
        self.title = title
        self.output_file = output_file
        self.fig = None
        self.ax = None
        self.trajectories: Dict[str, List[Vec3]] = {}
        self.colors: Dict[str, str] = {}
        self.update_interval = update_interval
        self.step_count = 0
        
    def initialize(self, objects: Dict[str, Body]) -> None:
        """Initialize the 3D plot."""
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(self.title)
        
        # Initialize trajectory storage
        for name, obj in objects.items():
            self.trajectories[name] = [obj.position]
            # Use object color if available, otherwise default colors
            if hasattr(obj, 'color') and obj.color:
                self.colors[name] = obj.color
            else:
                # Assign from default color palette
                self.colors[name] = self.DEFAULT_COLOR_PALETTE[len(self.colors) % len(self.DEFAULT_COLOR_PALETTE)]
    
    def update(self, objects: Dict[str, Body]) -> None:
        """Update trajectories with current positions."""
        self.step_count += 1
        for name, obj in objects.items():
            if name not in self.trajectories:
                self.trajectories[name] = []
                self.colors[name] = 'gray'
            self.trajectories[name].append(obj.position)
    
    def render(self, objects: Dict[str, Body], show_trajectories: bool = True, 
               max_trail_length: int = 1000) -> None:
        """Render the current state of the simulation."""
        if self.ax is None:
            self.initialize(objects)
        
        self.ax.clear()
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title(self.title)
        
        # Plot trajectories
        if show_trajectories:
            for name, trajectory in self.trajectories.items():
                if len(trajectory) > 1:
                    # Limit trajectory length for performance
                    recent_traj = trajectory[-max_trail_length:]
                    xs = [p[0] for p in recent_traj]
                    ys = [p[1] for p in recent_traj]
                    zs = [p[2] for p in recent_traj]
                    self.ax.plot(xs, ys, zs, color=self.colors.get(name, 'gray'), 
                               alpha=0.5, linewidth=1, label=f"{name} trail")
        
        # Plot current positions as spheres
        for name, obj in objects.items():
            x, y, z = obj.position
            color = self.colors.get(name, 'gray')
            # Size based on mass (logarithmic scale for better visibility)
            size = max(
                self.MIN_SPHERE_SIZE,
                min(self.MAX_SPHERE_SIZE, 
                    self.BASE_SPHERE_SIZE + self.SIZE_SCALE_FACTOR * math.log10(obj.mass + 1))
            )
            self.ax.scatter([x], [y], [z], color=color, s=size, 
                          marker='o', edgecolors='black', linewidth=0.5, label=name)
        
        # Auto-scale axes to fit all objects
        all_positions = [obj.position for obj in objects.values()]
        if all_positions:
            xs = [p[0] for p in all_positions]
            ys = [p[1] for p in all_positions]
            zs = [p[2] for p in all_positions]
            
            # Add some padding
            x_range = max(xs) - min(xs) if len(xs) > 1 else 1e6
            y_range = max(ys) - min(ys) if len(ys) > 1 else 1e6
            z_range = max(zs) - min(zs) if len(zs) > 1 else 1e6
            max_range = max(x_range, y_range, z_range) * 1.2
            
            mid_x = (max(xs) + min(xs)) / 2
            mid_y = (max(ys) + min(ys)) / 2
            mid_z = (max(zs) + min(zs)) / 2
            
            self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
            self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
            self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        self.ax.legend(loc='upper right', fontsize='small')
        plt.draw()
        plt.pause(0.001)
    
    def show(self) -> None:
        """Display the final plot."""
        if self.fig:
            plt.show()
    
    def save(self, filename: str | None = None) -> None:
        """Save the current plot to a file.
        
        Args:
            filename: Output filename. If None, uses self.output_file.
        """
        if self.fig:
            output = filename or self.output_file
            plt.savefig(output, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {output}")
    
    def create_animation(self, objects: Dict[str, Body], output_file: str = "gravity_animation.mp4",
                        fps: int = 30, show_trajectories: bool = True) -> None:
        """Create an animation from stored trajectories.
        
        Args:
            objects: Final state of objects (for metadata)
            output_file: Output filename (supports .mp4, .gif)
            fps: Frames per second
            show_trajectories: Whether to show trajectory trails
        """
        if not HAS_MATPLOTLIB:
            raise RuntimeError("matplotlib is required for animation")
        
        try:
            from matplotlib import animation
        except ImportError:
            raise RuntimeError("matplotlib.animation is required for creating animations")
        
        # Check if we have trajectory data
        if not self.trajectories or all(len(t) == 0 for t in self.trajectories.values()):
            print("⚠️  No trajectory data available for animation")
            return
        
        # Find the maximum trajectory length
        max_frames = max(len(traj) for traj in self.trajectories.values())
        if max_frames < 2:
            print("⚠️  Need at least 2 frames for animation")
            return
        
        print(f"📹 Creating animation with {max_frames} frames...")
        
        # Create figure for animation
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        def init():
            """Initialize animation."""
            self.ax.clear()
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(self.title)
            return []
        
        def animate(frame):
            """Animate a single frame."""
            self.ax.clear()
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_zlabel('Z (m)')
            self.ax.set_title(f"{self.title} - Frame {frame}/{max_frames}")
            
            # Collect all positions for this frame (and previous for trails)
            all_positions_frame = []
            
            # Plot trajectories
            if show_trajectories:
                for name, trajectory in self.trajectories.items():
                    if frame < len(trajectory):
                        # Show trail up to current frame
                        trail = trajectory[:frame+1]
                        if len(trail) > 1:
                            xs = [p[0] for p in trail]
                            ys = [p[1] for p in trail]
                            zs = [p[2] for p in trail]
                            self.ax.plot(xs, ys, zs, color=self.colors.get(name, 'gray'),
                                       alpha=0.5, linewidth=1)
            
            # Plot current positions
            for name, trajectory in self.trajectories.items():
                if frame < len(trajectory):
                    pos = trajectory[frame]
                    all_positions_frame.append(pos)
                    x, y, z = pos
                    color = self.colors.get(name, 'gray')
                    
                    # Get object mass from objects dict if available
                    mass = objects[name].mass if name in objects else 1e30
                    size = max(
                        self.MIN_SPHERE_SIZE,
                        min(self.MAX_SPHERE_SIZE,
                            self.BASE_SPHERE_SIZE + self.SIZE_SCALE_FACTOR * math.log10(mass + 1))
                    )
                    self.ax.scatter([x], [y], [z], color=color, s=size,
                                  marker='o', edgecolors='black', linewidth=0.5, label=name)
            
            # Auto-scale axes
            if all_positions_frame:
                xs = [p[0] for p in all_positions_frame]
                ys = [p[1] for p in all_positions_frame]
                zs = [p[2] for p in all_positions_frame]
                
                # Get all trajectory points for consistent scaling
                all_traj_points = []
                for trajectory in self.trajectories.values():
                    all_traj_points.extend(trajectory[:frame+1])
                
                if all_traj_points:
                    all_xs = [p[0] for p in all_traj_points]
                    all_ys = [p[1] for p in all_traj_points]
                    all_zs = [p[2] for p in all_traj_points]
                    
                    x_range = max(all_xs) - min(all_xs) if len(all_xs) > 1 else 1e6
                    y_range = max(all_ys) - min(all_ys) if len(all_ys) > 1 else 1e6
                    z_range = max(all_zs) - min(all_zs) if len(all_zs) > 1 else 1e6
                    max_range = max(x_range, y_range, z_range) * 1.2
                    
                    mid_x = (max(all_xs) + min(all_xs)) / 2
                    mid_y = (max(all_ys) + min(all_ys)) / 2
                    mid_z = (max(all_zs) + min(all_zs)) / 2
                    
                    self.ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
                    self.ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
                    self.ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
            
            self.ax.legend(loc='upper right', fontsize='small')
            return []
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, animate, init_func=init,
            frames=max_frames, interval=1000/fps, blit=False
        )
        
        # Save animation
        try:
            if output_file.endswith('.gif'):
                print("💾 Saving as GIF (this may take a while)...")
                anim.save(output_file, writer='pillow', fps=fps)
            elif output_file.endswith('.mp4'):
                print("💾 Saving as MP4...")
                try:
                    anim.save(output_file, writer='ffmpeg', fps=fps, extra_args=['-vcodec', 'libx264'])
                except Exception as e:
                    print(f"⚠️  FFmpeg not available, falling back to pillow writer")
                    # Try with pillow as fallback
                    gif_file = output_file.replace('.mp4', '.gif')
                    anim.save(gif_file, writer='pillow', fps=fps)
                    print(f"✅ Saved as GIF instead: {gif_file}")
                    return
            else:
                print(f"⚠️  Unsupported format, saving as GIF")
                gif_file = output_file.replace(Path(output_file).suffix, '.gif')
                anim.save(gif_file, writer='pillow', fps=fps)
                output_file = gif_file
            
            print(f"✅ Animation saved: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save animation: {e}")
            print("💡 Tip: Install ffmpeg for MP4 or pillow for GIF support")
            print("   pip install pillow")



class GravityLaw(Protocol):
    """Protocol for custom gravity laws."""
    def __call__(self, mass: float, distance: float, **kwargs) -> float:
        """Calculate acceleration magnitude given mass and distance.
        
        Args:
            mass: Mass of source body (kg)
            distance: Distance between bodies (m)
            **kwargs: Additional parameters (e.g., velocity for GR corrections)
            
        Returns:
            Acceleration magnitude (m/s^2)
        """
        ...


def newtonian_gravity(mass: float, distance: float, **kwargs) -> float:
    """Standard Newtonian gravity: a = G * M / r^2"""
    return G * mass / (distance ** 2)


def modified_newtonian_gravity(mass: float, distance: float, **kwargs) -> float:
    """Modified Newtonian Dynamics (MOND) - simplified example.
    
    At large distances, gravity falls off as 1/r instead of 1/r^2.
    Transition scale a0 ~ 1.2e-10 m/s^2
    """
    a0 = 1.2e-10  # MOND acceleration scale
    a_newton = G * mass / (distance ** 2)
    
    # Interpolation function: μ(x) ≈ x for x << 1, μ(x) ≈ 1 for x >> 1
    # Using simple form: μ(x) = x / (1 + x)
    x = a_newton / a0
    mu = x / (1.0 + x)
    
    return a_newton * mu


def schwarzschild_correction(mass: float, distance: float, **kwargs) -> float:
    """First-order General Relativity correction to Newtonian gravity.
    
    Includes 1PN (first post-Newtonian) correction term.
    For weak fields: a ≈ a_N * (1 + 3 * GM/(r*c^2))
    """
    c = 299792458.0  # Speed of light (m/s)
    a_newton = G * mass / (distance ** 2)
    
    # Schwarzschild radius: r_s = 2GM/c^2
    r_s = 2 * G * mass / (c ** 2)
    
    # First-order correction (simplified)
    correction = 1.0 + 1.5 * r_s / distance
    
    return a_newton * correction


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
    def __init__(self, gravity_law: GravityLaw | None = None) -> None:
        """Initialize with optional custom gravity law.
        
        Args:
            gravity_law: Custom gravity law function. Defaults to Newtonian.
        """
        self.gravity_law = gravity_law or newtonian_gravity
    def _accelerations_for_positions(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        positions: Dict[str, Vec3],
    ) -> Dict[str, Vec3]:
        accelerations: Dict[str, Vec3] = {name: (0.0, 0.0, 0.0) for name in objects}
        for source_name, target_name in pull_pairs:
            source = objects[source_name]
            target = objects[target_name]
            displacement = v_sub(positions[source_name], positions[target_name])
            r = max(v_mag(displacement), 1e-9)
            
            # Use custom gravity law
            acc_mag = self.gravity_law(source.mass, r)
            
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


class NumPyPhysicsBackend:
    """NumPy-accelerated physics backend for high-performance N-body simulations.
    
    This backend uses vectorized NumPy operations for significant performance
    improvements (10x-50x) when simulating many objects (100s-1000s of bodies).
    """
    
    def __init__(self, gravity_law: GravityLaw | None = None) -> None:
        """Initialize with optional custom gravity law.
        
        Args:
            gravity_law: Custom gravity law function. Defaults to Newtonian.
        
        Raises:
            RuntimeError: If NumPy is not installed.
        """
        if not HAS_NUMPY:
            raise RuntimeError(
                "NumPy is required for NumPyPhysicsBackend. "
                "Install it with: pip install numpy"
            )
        self.gravity_law = gravity_law or newtonian_gravity
    
    def _accelerations_for_positions_numpy(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        positions_dict: Dict[str, Vec3],
    ) -> Dict[str, Vec3]:
        """Calculate accelerations using vectorized NumPy operations."""
        if not pull_pairs:
            return {name: (0.0, 0.0, 0.0) for name in objects}
        
        # Build index mapping
        name_to_idx = {name: i for i, name in enumerate(objects.keys())}
        n = len(objects)
        
        # Initialize acceleration array
        acc_array = np.zeros((n, 3), dtype=np.float64)
        
        # Vectorize pull pairs
        source_indices = []
        target_indices = []
        masses = []
        
        for source_name, target_name in pull_pairs:
            source_indices.append(name_to_idx[source_name])
            target_indices.append(name_to_idx[target_name])
            masses.append(objects[source_name].mass)
        
        if not source_indices:
            return {name: (0.0, 0.0, 0.0) for name in objects}
        
        # Convert to numpy arrays
        source_idx = np.array(source_indices, dtype=np.int32)
        target_idx = np.array(target_indices, dtype=np.int32)
        mass_array = np.array(masses, dtype=np.float64)
        
        # Build position array
        pos_array = np.zeros((n, 3), dtype=np.float64)
        for name, idx in name_to_idx.items():
            pos = positions_dict[name]
            pos_array[idx] = [pos[0], pos[1], pos[2]]
        
        # Vectorized displacement calculation
        displacements = pos_array[source_idx] - pos_array[target_idx]
        
        # Vectorized distance calculation (with minimum threshold)
        distances = np.maximum(np.linalg.norm(displacements, axis=1), 1e-9)
        
        # Vectorized acceleration magnitude using custom gravity law
        # For performance, we vectorize the gravity law if possible
        if self.gravity_law == newtonian_gravity:
            # Fast path for Newtonian gravity
            acc_mags = G * mass_array / (distances ** 2)
        else:
            # Slower path for custom gravity laws (element-wise)
            acc_mags = np.array([
                self.gravity_law(m, d) for m, d in zip(mass_array, distances)
            ], dtype=np.float64)
        
        # Vectorized direction normalization
        directions = displacements / distances[:, np.newaxis]
        
        # Vectorized acceleration vectors
        acc_vectors = directions * acc_mags[:, np.newaxis]
        
        # Accumulate accelerations for each target
        for i, target_i in enumerate(target_idx):
            acc_array[target_i] += acc_vectors[i]
        
        # Convert back to dictionary
        result = {}
        for name, idx in name_to_idx.items():
            result[name] = (float(acc_array[idx, 0]), float(acc_array[idx, 1]), float(acc_array[idx, 2]))
        
        return result
    
    def _accelerations(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]]) -> Dict[str, Vec3]:
        positions = {name: body.position for name, body in objects.items()}
        return self._accelerations_for_positions_numpy(objects, pull_pairs, positions)
    
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

    def _step_verlet(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        """Velocity Verlet integrator."""
        accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            body.position = v_add(
                body.position,
                v_add(
                    v_scale(body.velocity, dt),
                    v_scale(accelerations[name], 0.5 * dt * dt)
                )
            )
        
        new_accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            body.velocity = v_add(
                body.velocity,
                v_scale(
                    v_add(accelerations[name], new_accelerations[name]),
                    0.5 * dt
                )
            )

    def _step_euler(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        """Simple Euler integrator."""
        accelerations = self._accelerations(objects, pull_pairs)
        
        for name, body in objects.items():
            if body.fixed:
                continue
            body.velocity = v_add(body.velocity, v_scale(accelerations[name], dt))
            body.position = v_add(body.position, v_scale(body.velocity, dt))

    def _step_rk4(self, objects: Dict[str, Body], pull_pairs: List[Tuple[str, str]], dt: float) -> None:
        """RK4 integrator - uses same logic as PythonPhysicsBackend but with NumPy accelerations."""
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

        acc1 = self._accelerations_for_positions_numpy(objects, pull_pairs, base_pos)
        k1_r = {name: base_vel[name] for name in movable}
        k1_v = {name: acc1[name] for name in movable}

        pos2 = shifted_positions(k1_r, dt * 0.5)
        vel2 = shifted_velocities(k1_v, dt * 0.5)
        acc2 = self._accelerations_for_positions_numpy(objects, pull_pairs, pos2)
        k2_r = {name: vel2[name] for name in movable}
        k2_v = {name: acc2[name] for name in movable}

        pos3 = shifted_positions(k2_r, dt * 0.5)
        vel3 = shifted_velocities(k2_v, dt * 0.5)
        acc3 = self._accelerations_for_positions_numpy(objects, pull_pairs, pos3)
        k3_r = {name: vel3[name] for name in movable}
        k3_v = {name: acc3[name] for name in movable}

        pos4 = shifted_positions(k3_r, dt)
        vel4 = shifted_velocities(k3_v, dt)
        acc4 = self._accelerations_for_positions_numpy(objects, pull_pairs, pos4)
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


class CppPhysicsBackend(NumPyPhysicsBackend):
    """C++-accelerated backend for pairwise acceleration accumulation.

    Uses a tiny native kernel (compiled with g++) to compute acceleration vectors
    and keeps integrators in Python for compatibility with existing DSL behavior.
    """

    def __init__(self, gravity_law: GravityLaw | None = None) -> None:
        super().__init__(gravity_law=gravity_law)
        if self.gravity_law != newtonian_gravity:
            raise RuntimeError("CppPhysicsBackend currently supports only Newtonian gravity")
        self._kernel = self._load_kernel()

    def _load_kernel(self) -> Any:
        gxx = shutil.which("g++")
        if not gxx:
            raise RuntimeError("g++ compiler not found. Install g++ or use --backend numpy")
        src = r'''#include <cmath>
extern "C" void compute_acc(
    int n,
    int pair_count,
    const int* src_idx,
    const int* dst_idx,
    const double* masses,
    const double* pos,
    double* acc
){
    const double G = 6.67430e-11;
    for(int i=0;i<n*3;i++) acc[i]=0.0;
    for(int p=0;p<pair_count;p++){
        int s = src_idx[p];
        int t = dst_idx[p];
        double dx = pos[s*3+0]-pos[t*3+0];
        double dy = pos[s*3+1]-pos[t*3+1];
        double dz = pos[s*3+2]-pos[t*3+2];
        double r = std::sqrt(dx*dx+dy*dy+dz*dz);
        if(r<1e-9) r=1e-9;
        double a = G*masses[p]/(r*r);
        double invr = 1.0/r;
        acc[t*3+0] += a*dx*invr;
        acc[t*3+1] += a*dy*invr;
        acc[t*3+2] += a*dz*invr;
    }
}
'''
        td = Path(tempfile.gettempdir()) / "gravity_lang_cpp_kernel"
        td.mkdir(parents=True, exist_ok=True)
        cpp = td / "kernel.cpp"
        so = td / ("kernel.dll" if sys.platform.startswith("win") else "kernel.so")
        cpp.write_text(src, encoding="utf-8")
        compile_cmd = [gxx, "-O3", "-shared", "-std=c++17"]
        if not sys.platform.startswith("win"):
            compile_cmd.append("-fPIC")
        compile_cmd += [str(cpp), "-o", str(so)]
        try:
            subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compile C++ kernel: {exc.stderr.strip()}") from exc

        lib = ctypes.CDLL(str(so))
        fn = lib.compute_acc
        fn.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        ]
        fn.restype = None
        return fn

    def _accelerations_for_positions_numpy(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        positions_dict: Dict[str, Vec3],
    ) -> Dict[str, Vec3]:
        if not pull_pairs:
            return {name: (0.0, 0.0, 0.0) for name in objects}

        name_to_idx = {name: i for i, name in enumerate(objects.keys())}
        n = len(objects)
        pair_count = len(pull_pairs)

        src = np.array([name_to_idx[s] for s, _ in pull_pairs], dtype=np.int32)
        dst = np.array([name_to_idx[t] for _, t in pull_pairs], dtype=np.int32)
        masses = np.array([objects[s].mass for s, _ in pull_pairs], dtype=np.float64)

        pos = np.zeros((n, 3), dtype=np.float64)
        for name, idx in name_to_idx.items():
            pos[idx] = positions_dict[name]
        acc = np.zeros((n, 3), dtype=np.float64)

        self._kernel(
            ctypes.c_int(n),
            ctypes.c_int(pair_count),
            src.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            dst.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            masses.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            acc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        return {
            name: (float(acc[idx, 0]), float(acc[idx, 1]), float(acc[idx, 2]))
            for name, idx in name_to_idx.items()
        }


class JuliaDiffEqBackend:
    """Julia DifferentialEquations.jl backend.

    This backend calls Julia to solve Newtonian N-body equations with
    DifferentialEquations.jl's Tsit5 solver for each step.
    """

    def __init__(self, julia_bin: str = "julia") -> None:
        self.julia_bin = julia_bin

    def _build_payload(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        dt: float,
    ) -> str:
        payload = {
            "dt": dt,
            "objects": {
                name: {
                    "position": list(body.position),
                    "velocity": list(body.velocity),
                    "mass": body.mass,
                    "fixed": body.fixed,
                }
                for name, body in objects.items()
            },
            "pull_pairs": pull_pairs,
        }
        return json.dumps(payload)

    def step(
        self,
        objects: Dict[str, Body],
        pull_pairs: List[Tuple[str, str]],
        dt: float,
        integrator: Integrator,
    ) -> None:
        _ = integrator  # DifferentialEquations.jl controls its own solver strategy.
        script = r'''
using JSON
using DifferentialEquations

function solve_step(payload)
    names = collect(keys(payload["objects"]))
    sort!(names)
    index = Dict(name => i for (i, name) in enumerate(names))
    n = length(names)

    x0 = zeros(Float64, 6n)
    masses = zeros(Float64, n)
    fixed = falses(n)

    for (i, name) in enumerate(names)
        body = payload["objects"][name]
        x0[3i-2:3i] .= Float64.(body["position"])
        x0[3n+3i-2:3n+3i] .= Float64.(body["velocity"])
        masses[i] = body["mass"]
        fixed[i] = body["fixed"]
    end

    pairs = [(index[p[1]], index[p[2]]) for p in payload["pull_pairs"]]
    G = 6.67430e-11

    function rhs!(du, u, p, t)
        pos = reshape(view(u, 1:3n), 3, n)
        vel = reshape(view(u, 3n+1:6n), 3, n)

        du[1:3n] .= vec(vel)
        du[3n+1:6n] .= 0.0

        for (src, dst) in pairs
            if fixed[dst]
                continue
            end
            dx = pos[1, src] - pos[1, dst]
            dy = pos[2, src] - pos[2, dst]
            dz = pos[3, src] - pos[3, dst]
            r = max(sqrt(dx*dx + dy*dy + dz*dz), 1e-9)
            a = G * masses[src] / (r * r)
            base = 3n + 3dst - 2
            du[base] += a * dx / r
            du[base + 1] += a * dy / r
            du[base + 2] += a * dz / r
        end
    end

    prob = ODEProblem(rhs!, x0, (0.0, payload["dt"]))
    sol = solve(prob, Tsit5(); abstol=1e-9, reltol=1e-9)
    y = sol.u[end]

    out = Dict{String, Any}()
    for (i, name) in enumerate(names)
        out[name] = Dict(
            "position" => [y[3i-2], y[3i-1], y[3i]],
            "velocity" => [y[3n+3i-2], y[3n+3i-1], y[3n+3i]],
        )
    end
    println(JSON.json(out))
end

payload = JSON.parse(read(stdin, String))
solve_step(payload)
'''
        payload = self._build_payload(objects, pull_pairs, dt)
        try:
            result = subprocess.run(
                [self.julia_bin, "-e", script],
                input=payload,
                text=True,
                capture_output=True,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Julia binary not found. Install Julia and DifferentialEquations.jl "
                "or choose --backend python/--backend numpy."
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if exc.stderr else "Unknown Julia backend failure"
            raise RuntimeError(
                "Julia DifferentialEquations backend failed. Ensure packages JSON and "
                f"DifferentialEquations are installed. Details: {stderr}"
            ) from exc

        updated = json.loads(result.stdout)
        for name, state in updated.items():
            objects[name].position = tuple(state["position"])  # type: ignore[assignment]
            objects[name].velocity = tuple(state["velocity"])  # type: ignore[assignment]


def _default_backend_name(julia_bin: str = "julia") -> BackendName:
    """Prefer fastest available local backend with graceful fallback."""
    if HAS_NUMPY:
        try:
            _ = CppPhysicsBackend()
            return "cpp"
        except RuntimeError:
            pass
    if HAS_NUMPY:
        return "numpy"
    if shutil.which(julia_bin):
        return "julia_diffeq"
    return "python"



class GravityInterpreter:
    def __init__(self, physics_backend: PhysicsBackend | None = None, enable_3d_viz: bool = False, 
                 viz_interval: int = 1) -> None:
        self.objects: Dict[str, Body] = {}
        self.pull_pairs: set[Tuple[str, str]] = set()  # Using set for O(1) membership checks
        self.output: List[str] = []
        self.observers: List[Observer] = []
        self.variables: Dict[str, float] = {}
        self.global_friction = 0.0
        self.enable_collisions = True
        self.physics_backend = physics_backend or create_physics_backend("auto")
        self.visualizer: Visualizer3D | None = None
        self.enable_3d_viz = enable_3d_viz
        if enable_3d_viz:
            self.visualizer = Visualizer3D("Gravity-Lang 3D Simulation", 
                                          update_interval=viz_interval)

    def _format_float(self, value: float) -> str:
        return f"{value:.6e}"

    def parse_value(self, token: str) -> float:
        token = token.strip()
        if token in self.variables:
            return self.variables[token]
        m = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)(?:\[([a-zA-Z]+)\])?", token)
        if not m:
            raise ParseError(
                f"Invalid numeric value: '{token}'",
                "Use format: number[unit] (e.g., '100[km]', '5.5[s]', '1e6[kg]')"
            )
        value = float(m.group(1))
        unit = m.group(2)
        if unit:
            if unit not in UNIT_SCALE:
                available_units = ", ".join(UNIT_SCALE.keys())
                raise UnitError(
                    f"Unknown unit: '{unit}'",
                    f"Available units: {available_units}"
                )
            value *= UNIT_SCALE[unit]
        return value

    def parse_vector(self, token: str) -> Vec3:
        token = token.strip()
        m = re.fullmatch(r"(\[[^\]]+\])(?:\[([a-zA-Z/]+)\])?", token)
        if not m:
            raise ParseError(
                f"Invalid vector: '{token}'",
                "Use format: [x,y,z] or [x,y,z][unit] (e.g., '[1,2,3]', '[0,1,0][km/s]')"
            )

        vector_text = m.group(1)
        unit = m.group(2)
        inner = vector_text[1:-1]
        parts = [part.strip() for part in inner.split(",")]
        if len(parts) != 3:
            raise ParseError(
                f"Vector must have exactly 3 components, got {len(parts)}: '{token}'",
                "Use format: [x,y,z] with three comma-separated values"
            )

        if unit:
            if unit not in VECTOR_UNIT_SCALE:
                available_units = ", ".join(VECTOR_UNIT_SCALE.keys())
                raise UnitError(
                    f"Unknown vector unit: '{unit}'",
                    f"Available vector units: {available_units}"
                )
            if any("[" in part for part in parts):
                raise ParseError(
                    "Cannot use units on individual components and vector suffix",
                    "Use either [1[km],2[km],3[km]] OR [1,2,3][km]"
                )
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
            available = list(self.objects.keys())
            if available:
                suggestion = f"Available objects: {', '.join(available[:5])}"
                if len(available) > 5:
                    suggestion += f" (and {len(available) - 5} more)"
            else:
                suggestion = "No objects have been created yet. Create objects first with 'sphere', 'cube', etc."
            raise ObjectError(
                f"Object '{obj_name}' does not exist (referenced in {context})",
                suggestion
            )
        return self.objects[obj_name]

    def _parse_variable_assignment(self, line: str) -> None:
        m = re.fullmatch(r"let\s+(\w+)\s*=\s*(.+)", line)
        if not m:
            raise ParseError(
                f"Invalid let assignment: '{line}'",
                "Use format: let variable = value"
            )
        name = m.group(1)
        value = self.parse_value(m.group(2).strip())
        self.variables[name] = value

    def _evaluate_condition(self, expr: str) -> bool:
        m = re.fullmatch(r"(.+?)\s*(==|!=|<=|>=|<|>)\s*(.+)", expr.strip())
        if not m:
            raise ParseError(
                f"Invalid condition: '{expr}'",
                "Use format like: if x > 10 then print Earth.position"
            )
        left = self.parse_value(m.group(1).strip())
        op = m.group(2)
        right = self.parse_value(m.group(3).strip())
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "<":
            return left < right
        return left > right

    def _run_single_statement(self, line: str) -> bool:
        if line.startswith("let "):
            self._parse_variable_assignment(line)
            return True
        if line.startswith("if ") and " then " in line:
            cond_text, stmt = line[3:].split(" then ", 1)
            if self._evaluate_condition(cond_text):
                nested = stmt.strip()
                if self._run_single_statement(nested):
                    return True
                if nested.startswith("thrust "):
                    self._parse_thrust(nested)
                    return True
                if nested.startswith("print "):
                    self._exec_print(nested)
                    return True
                if nested == "monitor energy":
                    self._exec_monitor_energy()
                    return True
                if nested.endswith(".velocity") and "=" in nested:
                    self._parse_velocity_assignment(nested)
                    return True
            return True
        return False

    def execute(self, source: str) -> List[str]:
        lines = []
        for raw_line in source.splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if line:
                lines.append(line)
        i = 0
        while i < len(lines):
            line = lines[i]
            if self._run_single_statement(line):
                i += 1
            elif line.startswith(("sphere ", "cube ", "pointmass ", "probe ")):
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
                source_name = a.strip()
                # Split by comma and strip whitespace to support multiple objects
                target_names = [t.strip() for t in b.split(",")]
                self._require_object(source_name, "pull statement")
                for target_name in target_names:
                    self._require_object(target_name, "pull statement")
                    self.pull_pairs.add((source_name, target_name))
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
                raise ParseError(
                    f"Unknown statement: '{line}'",
                    "Valid statements: sphere/cube/pointmass/probe, pull, grav all, friction, collisions, thrust, simulate, orbit, print, monitor energy, observe, orbital_elements"
                )
        return self.output

    def _parse_object(self, line: str) -> None:
        try:
            shape, rest = line.split(" ", 1)
            name, rest = rest.split(" at ", 1)
        except ValueError as exc:
            raise ParseError(
                f"Invalid object syntax: '{line}'",
                "Use format: sphere ObjectName at [x,y,z] mass value[unit] ...",
                line_number=None
            ) from exc

        # Check for duplicate object names
        if name.strip() in self.objects:
            raise ObjectError(
                f"Object '{name.strip()}' already exists",
                f"Each object must have a unique name. Choose a different name or remove the existing '{name.strip()}' object."
            )

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
            raise ParseError(
                f"Object declaration missing mass: '{line}'",
                "Add 'mass value[unit]' (e.g., 'mass 5.972e24[kg]')"
            )
        mass = self.parse_value(m_mass.group(1))
        
        # Validate positive mass
        if mass <= 0:
            raise ValidationError(
                f"Mass must be positive, got {mass}",
                "Physical objects cannot have zero or negative mass. Use a positive value like '1[kg]' or '1e30[kg]'"
            )

        radius = 1.0
        velocity: Vec3 = (0.0, 0.0, 0.0)
        fixed = "fixed" in trailing

        m_radius = re.search(r"radius\s+([^\s]+)", trailing)
        if m_radius:
            radius = self.parse_value(m_radius.group(1))
            # Validate positive radius
            if radius <= 0:
                raise ValidationError(
                    f"Radius must be positive, got {radius}",
                    "Use a positive value like '6371[km]' for Earth-sized objects"
                )

        if "velocity" in trailing:
            vel_tail = trailing.split("velocity", 1)[1].strip()
            velocity_vector, vel_remaining = self._split_leading_vector(vel_tail)
            if vel_remaining.startswith("["):
                vel_unit, _ = self._split_leading_vector(vel_remaining)
                velocity_vector = f"{velocity_vector}{vel_unit}"
            velocity = self.parse_vector(velocity_vector)

        # Parse optional color attribute
        # Note: Color validation is deferred to matplotlib when rendering.
        # Accepts matplotlib color names ('blue', 'red', etc.) or hex codes ('#0000FF').
        # Invalid colors will cause matplotlib to raise an error during visualization.
        color = ""
        m_color = re.search(r'color\s+"([^"]+)"', trailing)
        if m_color:
            color = m_color.group(1)

        self.objects[name.strip()] = Body(
            name=name.strip(),
            shape=shape,
            position=position,
            velocity=velocity,
            radius=radius,
            mass=mass,
            fixed=fixed,
            color=color,
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
                self.pull_pairs.add(pair_a)
                self.pull_pairs.add(pair_b)

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

    def _copy_objects_state(self, objects: Dict[str, Body]) -> Dict[str, Body]:
        return {
            name: Body(
                name=body.name,
                shape=body.shape,
                position=body.position,
                velocity=body.velocity,
                mass=body.mass,
                radius=body.radius,
                fixed=body.fixed,
                color=body.color,
                properties=dict(body.properties),
            )
            for name, body in objects.items()
        }

    def _assign_objects_state(self, target: Dict[str, Body], source: Dict[str, Body]) -> None:
        for name in target:
            target[name].position = source[name].position
            target[name].velocity = source[name].velocity

    def _estimate_state_error(self, a: Dict[str, Body], b: Dict[str, Body]) -> float:
        max_error = 0.0
        for name, body_a in a.items():
            body_b = b[name]
            if body_a.fixed:
                continue
            pos_scale = max(v_mag(body_a.position), v_mag(body_b.position), 1.0)
            vel_scale = max(v_mag(body_a.velocity), v_mag(body_b.velocity), 1.0)
            pos_err = v_mag(v_sub(body_a.position, body_b.position)) / pos_scale
            vel_err = v_mag(v_sub(body_a.velocity, body_b.velocity)) / vel_scale
            max_error = max(max_error, pos_err, vel_err)
        return max_error

    def _adaptive_integrate_step(
        self,
        step_pairs: List[Tuple[str, str]],
        dt: float,
        integrator: Integrator,
        adaptive_tol: float,
        adaptive_min_dt: float,
        adaptive_max_dt: float,
        next_dt_hint: float,
    ) -> float:
        remaining = dt
        trial_dt = min(max(next_dt_hint, adaptive_min_dt), adaptive_max_dt, dt)

        while remaining > 1e-12:
            current_dt = min(trial_dt, remaining)
            if current_dt < adaptive_min_dt:
                current_dt = min(adaptive_min_dt, remaining)

            start_state = self._copy_objects_state(self.objects)
            single_step_state = self._copy_objects_state(start_state)
            self.physics_backend.step(single_step_state, step_pairs, current_dt, integrator)

            two_half_state = self._copy_objects_state(start_state)
            half_dt = current_dt * 0.5
            self.physics_backend.step(two_half_state, step_pairs, half_dt, integrator)
            self.physics_backend.step(two_half_state, step_pairs, half_dt, integrator)

            relative_error = self._estimate_state_error(single_step_state, two_half_state)

            if relative_error <= adaptive_tol or current_dt <= adaptive_min_dt * 1.001:
                self._assign_objects_state(self.objects, two_half_state)
                self._apply_friction(current_dt)
                self._resolve_collisions()
                remaining -= current_dt

                if relative_error <= 1e-16:
                    growth = 2.0
                else:
                    growth = min(2.0, max(1.1, 0.9 * (adaptive_tol / relative_error) ** 0.2))
                trial_dt = min(adaptive_max_dt, max(adaptive_min_dt, current_dt * growth))
                continue

            shrink = max(0.2, 0.9 * (adaptive_tol / max(relative_error, 1e-16)) ** 0.25)
            trial_dt = max(adaptive_min_dt, current_dt * shrink)

        return trial_dt

    def _run_loop(self, lines: List[str], start: int) -> int:
        header = lines[start]
        # Updated regex to support optional units on range values: 0..100[days]
        adaptive_clause = r"(?:\s+adaptive(?:\s+tol\s+([^\s]+))?(?:\s+min\s+([^\s]+))?(?:\s+max\s+([^\s]+))?)?"
        m_orbit = re.fullmatch(
            rf"orbit\s+\w+\s+in\s+(\d+(?:\[[^\]]+\])?)\.\.(\d+(?:\[[^\]]+\])?)\s+dt\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4|verlet|euler))?{adaptive_clause}\s*\{{",
            header,
        )
        m_sim = re.fullmatch(
            rf"simulate\s+\w+\s+in\s+(\d+(?:\[[^\]]+\])?)\.\.(\d+(?:\[[^\]]+\])?)\s+step\s+([^\s]+)(?:\s+integrator\s+(leapfrog|rk4|verlet|euler))?{adaptive_clause}\s*\{{",
            header,
        )
        match = m_orbit or m_sim
        if not match:
            raise ValueError(f"Invalid loop statement: {header}")

        # Parse range values which may include units
        begin_str = match.group(1)
        end_str = match.group(2)
        begin = int(self.parse_value(begin_str))
        end = int(self.parse_value(end_str))
        dt = self.parse_value(match.group(3))
        integrator: Integrator = (match.group(4) or "leapfrog").lower()  # type: ignore[assignment]
        adaptive_enabled = " adaptive" in header
        adaptive_tol = float(match.group(5) or "1e-6")
        adaptive_min_dt = self.parse_value(match.group(6) or "0.01[s]")
        adaptive_max_dt = self.parse_value(match.group(7) or match.group(3))
        if adaptive_min_dt <= 0 or adaptive_max_dt <= 0:
            raise ValueError("Adaptive min/max dt must be positive")
        if adaptive_min_dt > adaptive_max_dt:
            raise ValueError("Adaptive min dt cannot be larger than max dt")
        next_dt_hint = min(max(adaptive_min_dt, dt), adaptive_max_dt)

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
                if self._run_single_statement(stmt):
                    continue
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
                    source_name = a.strip()
                    # Split by comma and strip whitespace to support multiple objects
                    target_names = [t.strip() for t in b.split(",")]
                    self._require_object(source_name, "pull statement")
                    for target_name in target_names:
                        self._require_object(target_name, "pull statement")
                        pair = (source_name, target_name)
                        self.pull_pairs.add(pair)
                        if pair not in step_pairs:
                            step_pairs.append(pair)

            # Execute physics step
            if not has_explicit_step:
                if adaptive_enabled:
                    next_dt_hint = self._adaptive_integrate_step(
                        step_pairs,
                        dt,
                        integrator,
                        adaptive_tol,
                        adaptive_min_dt,
                        adaptive_max_dt,
                        next_dt_hint,
                    )
                else:
                    self.physics_backend.step(self.objects, step_pairs, dt, integrator)
                    self._apply_friction(dt)
                    self._resolve_collisions()

            # Process step_physics and other statements
            for stmt in block:
                if self._run_single_statement(stmt):
                    continue
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
            
            # Update 3D visualization if enabled
            if self.visualizer:
                if step_index == 0:
                    self.visualizer.initialize(self.objects)
                self.visualizer.update(self.objects)
                # Render every N steps to avoid slowdown
                if step_index % self.visualizer.update_interval == 0:
                    self.visualizer.render(self.objects)
            
            step_index += 1
        
        # Show final visualization if enabled
        if self.visualizer:
            self.visualizer.render(self.objects)
            self.visualizer.save()  # Uses self.output_file from constructor
        
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



def get_backend_availability(julia_bin: str = "julia") -> Dict[str, str]:
    status: Dict[str, str] = {"python": "available"}
    status["numpy"] = "available" if HAS_NUMPY else "missing dependency: numpy"

    if HAS_NUMPY:
        try:
            _ = CppPhysicsBackend()
            status["cpp"] = "available"
        except RuntimeError as exc:
            status["cpp"] = f"unavailable: {exc}"
    else:
        status["cpp"] = "unavailable: requires numpy and g++"

    julia = shutil.which(julia_bin)
    status["julia_diffeq"] = "available" if julia else f"unavailable: julia binary not found at '{julia_bin}'"
    status["auto"] = f"selected={_default_backend_name(julia_bin=julia_bin)}"
    return status

def create_physics_backend(name: BackendName, julia_bin: str = "julia") -> PhysicsBackend:
    if name == "auto":
        return create_physics_backend(_default_backend_name(julia_bin=julia_bin), julia_bin=julia_bin)
    if name == "python":
        return PythonPhysicsBackend()
    if name == "numpy":
        return NumPyPhysicsBackend()
    if name == "cpp":
        return CppPhysicsBackend()
    if name == "julia_diffeq":
        return JuliaDiffEqBackend(julia_bin=julia_bin)
    raise ValueError(f"Unknown backend: {name}")


def run_script_file(script_path: str, enable_3d: bool = False, viz_interval: int = 1, 
                    create_animation: bool = False, anim_fps: int = 30,
                    backend_name: BackendName = "auto", julia_bin: str = "julia",
                    show_visualization: bool = True) -> List[str]:
    script = Path(script_path).read_text(encoding="utf-8")
    interpreter = GravityInterpreter(
        physics_backend=create_physics_backend(backend_name, julia_bin=julia_bin),
        enable_3d_viz=enable_3d,
        viz_interval=viz_interval,
    )
    output = interpreter.execute(script)
    
    if enable_3d and interpreter.visualizer:
        # Create animation if requested
        if create_animation:
            anim_filename = "gravity_animation.gif"
            print(f"\n🎬 Creating animation...")
            interpreter.visualizer.create_animation(
                interpreter.objects, 
                output_file=anim_filename,
                fps=anim_fps
            )
        
        # Show final visualization (skip in headless/non-interactive mode)
        if show_visualization:
            interpreter.visualizer.show()  # Keep window open at end
    
    return output


def check_script_file(script_path: str, backend_name: BackendName = "auto", julia_bin: str = "julia") -> None:
    run_script_file(script_path, backend_name=backend_name, julia_bin=julia_bin)


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
        "--hidden-import",
        "matplotlib",
        "--hidden-import",
        "PIL",
        "--collect-data",
        "matplotlib",
        "--collect-data",
        "PIL",
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


def benchmark_backends(
    object_count: int = 200,
    steps: int = 20,
    dt: float = 1.0,
    repeats: int = 3,
    warmup_steps: int = 2,
    backend_names: List[str] | None = None,
) -> Dict[str, float]:
    if object_count < 2:
        raise ValueError("object_count must be >= 2")
    if steps < 1 or repeats < 1 or warmup_steps < 0:
        raise ValueError("steps/repeats must be >= 1 and warmup_steps must be >= 0")

    def build_case() -> tuple[Dict[str, Body], List[Tuple[str, str]]]:
        objects: Dict[str, Body] = {}
        for i in range(object_count):
            objects[f"B{i}"] = Body(
                name=f"B{i}",
                shape="pointmass",
                position=(float(i * 1000), float((i % 13) * 300), float((i % 7) * 200)),
                velocity=(0.0, 0.0, 0.0),
                mass=5.0e20 + i * 1.0e17,
                radius=1.0,
            )
        pairs = [(f"B{i}", f"B{j}") for i in range(object_count) for j in range(object_count) if i != j]
        return objects, pairs

    selected = backend_names or ["python", "numpy", "cpp"]
    backends: Dict[str, PhysicsBackend] = {}
    for name in selected:
        try:
            backends[name] = create_physics_backend(name)  # type: ignore[arg-type]
        except Exception:
            continue

    if "python" not in backends:
        backends["python"] = PythonPhysicsBackend()

    timings: Dict[str, float] = {}
    for name, backend in backends.items():
        objs, pairs = build_case()

        for _ in range(warmup_steps):
            backend.step(objs, pairs, dt, "verlet")

        samples: List[float] = []
        for _ in range(repeats):
            objs, pairs = build_case()
            start = time.perf_counter()
            for _ in range(steps):
                backend.step(objs, pairs, dt, "verlet")
            samples.append((time.perf_counter() - start) * 1000.0)

        timings[name] = statistics.median(samples)
        timings[f"{name}_mean_ms"] = statistics.mean(samples)

    baseline = timings.get("python", 1.0)
    for name, elapsed in list(timings.items()):
        if name in backends and name != "python":
            timings[f"{name}_speedup"] = baseline / max(elapsed, 1e-9)
    return timings


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Gravity Lang CLI (v1.0)")
    parser.add_argument("--version", action="version", version=f"Gravity Lang {GRAVITY_LANG_VERSION}")

    sub = parser.add_subparsers(dest="command")

    run_parser = sub.add_parser("run", help="Run a .gravity script")
    run_parser.add_argument("run_file", help="Path to a .gravity script")
    run_parser.add_argument("--3d", dest="enable_3d", action="store_true", 
                           help="Enable 3D visualization (requires matplotlib)")
    run_parser.add_argument("--viz-interval", type=int, default=1, 
                           help="Visualization update interval (render every N steps)")
    run_parser.add_argument("--animate", dest="create_animation", action="store_true",
                           help="Create animation from simulation (requires --3d)")
    run_parser.add_argument("--fps", type=int, default=30,
                           help="Animation frames per second (default: 30)")
    run_parser.add_argument("--backend", choices=["auto", "python", "numpy", "cpp", "julia_diffeq"], default="auto",
                           help="Physics backend: auto (default prefers cpp), python, numpy, cpp, or julia_diffeq")
    run_parser.add_argument("--julia-bin", default="julia",
                           help="Julia binary path when using --backend julia_diffeq")
    run_parser.add_argument("--headless", action="store_true",
                           help="Run visualization/animation generation without opening window")

    check_parser = sub.add_parser("check", help="Parse and validate a .gravity script")
    check_parser.add_argument("check_file", help="Path to a .gravity script")
    check_parser.add_argument("--backend", choices=["auto", "python", "numpy", "cpp", "julia_diffeq"], default="auto",
                              help="Physics backend used for validation run")
    check_parser.add_argument("--julia-bin", default="julia",
                              help="Julia binary path when using --backend julia_diffeq")

    bench_parser = sub.add_parser("benchmark", help="Benchmark physics backends")
    bench_parser.add_argument("--objects", type=int, default=200, help="Number of objects")
    bench_parser.add_argument("--steps", type=int, default=20, help="Number of integration steps")
    bench_parser.add_argument("--dt", type=float, default=1.0, help="Step size in seconds")
    bench_parser.add_argument("--repeats", type=int, default=3, help="Benchmark repeats (median reported)")
    bench_parser.add_argument("--warmup", type=int, default=2, help="Warmup steps before timing")
    bench_parser.add_argument("--backends", default="python,numpy,cpp", help="Comma-separated backends to test")
    bench_parser.add_argument("--csv-out", default="", help="Optional path to write benchmark CSV")

    backends_parser = sub.add_parser("backends", help="Show backend availability on this machine")
    backends_parser.add_argument("--julia-bin", default="julia", help="Julia binary path")

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
        # Validate animation requires 3D
        if hasattr(args, 'create_animation') and args.create_animation and not args.enable_3d:
            print("⚠️  Warning: --animate requires --3d flag. Enabling 3D visualization.")
            args.enable_3d = True
        
        output = run_script_file(
            args.run_file, 
            enable_3d=args.enable_3d,
            viz_interval=args.viz_interval,
            create_animation=getattr(args, 'create_animation', False),
            anim_fps=getattr(args, 'fps', 30),
            backend_name=getattr(args, 'backend', 'python'),
            julia_bin=getattr(args, 'julia_bin', 'julia'),
            show_visualization=not getattr(args, 'headless', False),
        )
        print("\n".join(output))
        return 0

    if args.command == "check":
        check_script_file(
            args.check_file,
            backend_name=getattr(args, 'backend', 'python'),
            julia_bin=getattr(args, 'julia_bin', 'julia'),
        )
        print(f"OK: {args.check_file}")
        return 0

    if args.command == "benchmark":
        requested = [name.strip() for name in args.backends.split(",") if name.strip()]
        results = benchmark_backends(
            object_count=args.objects,
            steps=args.steps,
            dt=args.dt,
            repeats=args.repeats,
            warmup_steps=args.warmup,
            backend_names=requested,
        )
        print("metric,value")
        for key, value in results.items():
            print(f"{key},{value:.6f}")
        if args.csv_out:
            out = Path(args.csv_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                f.write("metric,value\n")
                for key, value in results.items():
                    f.write(f"{key},{value:.6f}\n")
            print(f"Saved benchmark CSV: {out}")
        return 0

    if args.command == "backends":
        statuses = get_backend_availability(julia_bin=args.julia_bin)
        print("backend,status")
        for name, status in statuses.items():
            print(f"{name},{status}")
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
        output = run_script_file(args.legacy_file, backend_name="auto")
        print("\n".join(output))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
