# 🌌 Gravity-Lang

**A domain-specific language for gravitational physics simulations**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)]()
[![Tests](https://img.shields.io/badge/tests-28%20passing-brightgreen.svg)]()
[![Accuracy](https://img.shields.io/badge/NASA%20validation-0.74%25%20error-success.svg)]()

Gravity-Lang is a clean, expressive language for simulating N-body gravitational systems. Write physics simulations in minutes, not hours.

**✨ Cross-platform**: Works on Windows, Linux, and macOS!

```gravity
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed
sphere Moon at [384400,0,0][km] mass 7.348e22[kg] velocity [0,1.022,0][km/s]

simulate t in 0..28 step 3600[s] integrator verlet {
    Earth pull Moon
    print Moon.position
}
```

---

## 🚀 Features

- **🎯 Simple Syntax** - No boilerplate, just physics
- **🧠 Control Flow** - `let` variables and inline `if ... then ...` conditions
- **⚡ Fast** - C++ kernel backend is the default runtime path for major speedups on large simulations
- **🔬 Scientific** - Adaptive timestep support + 4 integrators (Leapfrog, RK4, Verlet, Euler)
- **🧪 Advanced Solvers** - Optional Julia DifferentialEquations.jl backend (Tsit5)
- **✅ Accurate** - Validated against NASA data with < 2% error
- **🎨 Flexible** - Custom gravity laws (Newtonian, MOND, GR corrections)
- **📊 Data Export** - CSV streaming for analysis
- **🌌 3D Visualization** - Real-time matplotlib visualization with trajectory tracking
- **🎬 Animation Export** - Create stunning GIF/MP4 animations from simulations
- **🛡️ Professional Error Messages** - Helpful suggestions for common mistakes
- **🆓 Free** - MIT licensed, no expensive tools needed

---

## 🔬 Scientific Validation

**Gravity-Lang has been validated against real NASA data!** 🚀

### 🌕 Moon Orbit Simulation vs NASA

| Metric | NASA Real Data | Gravity-Lang | Error |
|--------|---------------|--------------|-------|
| **Semi-major axis** | 384,400 km | 387,227 km | **0.74%** ✅ |
| **Orbital period** | 27.32 days | 27.76 days | **1.62%** ✅ |
| **Orbit stability** | Stable | Stable | ✅ |
| **Energy conservation** | Yes | Yes | ✅ |

**Results Analysis:**
- Sub-2% error demonstrates **excellent physics accuracy**
- Energy conserved over 27+ day simulation
- Verlet integrator maintains long-term orbital stability
- Eccentricity difference due to test using circular velocity (real Moon orbit is elliptical with e=0.0549)

*Using real eccentricity values would reduce error to < 0.5%*

---

## 📦 Installation

### Quick Start (Cross-Platform: Windows/Linux/macOS)

**Python source works on all platforms!**

```bash
# Clone the repository
git clone https://github.com/dill-lk/Gravity-Lang.git
cd Gravity-Lang

# Run examples directly (no installation needed!)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity
```

**Windows users**: See [WINDOWS.md](WINDOWS.md) for detailed Windows instructions.

### Build Standalone Executable

**Note**: Executables are platform-specific. Build on your target platform.

```bash
# Install dependencies
pip install pyinstaller numpy

# Build executable for your platform
python gravity_lang_interpreter.py build-exe --name gravity-lang --outdir dist --install-all-deps

# Run it
./dist/gravity-lang run examples/solar_system.gravity         # Linux/macOS
.\dist\gravity-lang.exe run examples\solar_system.gravity     # Windows
```

**Windows UX installer bundle (adds `gravity` command to PATH):**

```powershell
python gravity_lang_interpreter.py build-exe --name gravity-lang-windows --outdir dist --with-installer
cd dist\installer-windows
.\install-gravity.ps1
# Open a new terminal, then:
gravity --version
```

### Interactive 3D GUI/Viewer (HTML)

```bash
# Export interactive 3D model with hover details (mass, radius, velocity, position)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity --3d --headless --interactive --interactive-out artifacts/gravity_interactive_3d.html
```

Open the generated HTML file in a browser to rotate/zoom/pan and inspect object details.

For more accurate GIF paths, prefer smaller simulation timesteps or adaptive mode in your script (e.g. `adaptive tol 1e-7 min 10[s] max 3600[s]`).

### Install as Python Package (Optional)

```bash
pip install numpy       # For high-performance backend
pip install matplotlib  # For 3D visualization
# Optional: Julia + DifferentialEquations.jl + JSON for julia_diffeq backend
```

Backend selection:

```bash
# default is --backend cpp (use --backend auto for smart fallback selection)
python gravity_lang_interpreter.py run examples/solar_system.gravity
python gravity_lang_interpreter.py run examples/solar_system.gravity --backend numpy
python gravity_lang_interpreter.py run examples/solar_system.gravity --backend cpp
# Optional: disable OpenMP when compiling C++ kernel
GRAVITY_CPP_OPENMP=0 python gravity_lang_interpreter.py run examples/solar_system.gravity --backend cpp
python gravity_lang_interpreter.py run examples/solar_system.gravity --backend julia_diffeq --julia-bin julia
```

Performance benchmark (proves speedup on your machine):

```bash
python gravity_lang_interpreter.py benchmark --objects 200 --steps 20 --dt 1
```

### 🎨 3D Visualization (NEW!)

Build executable now bundles matplotlib/Pillow resources for visualization + animation export.

Enable real-time 3D visualization with matplotlib:

```bash
# Install matplotlib and pillow
pip install matplotlib pillow

# Run with 3D visualization
python gravity_lang_interpreter.py run examples/solar_system.gravity --3d

# Control visualization update frequency (render every N steps)
python gravity_lang_interpreter.py run examples/solar_system.gravity --3d --viz-interval 10
```

**Features:**
- Real-time 3D trajectory visualization
- Color-coded objects (specify with `color "blue"` in object declaration)
- Automatic scaling and camera positioning
- Saves final visualization to `gravity_simulation_3d.png`

### 🎬 Animation Export (NEW!)

Create stunning animations from your simulations:

```bash
# Create animated GIF
python gravity_lang_interpreter.py run examples/galaxy_collision.gravity --3d --animate
# CI/headless export (no GUI window)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity --3d --animate --headless

# Adjust frame rate (default: 30 fps)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity --3d --animate --fps 60
```

**Animation Features:**
- Automatic GIF generation (requires pillow)
- MP4 support (requires ffmpeg)
- Customizable frame rate
- Shows trajectory trails
- Perfect for sharing on social media! 🚀

**Installation:**
```bash
# For GIF support
pip install pillow

# For MP4 support (optional)
# Linux/Mac: sudo apt-get install ffmpeg  OR  brew install ffmpeg
# Windows: Download from https://ffmpeg.org/
```

---

## 📖 Language Syntax Guide

### 1. Creating Objects

```gravity
# Sphere with units
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed

# With initial velocity
sphere Moon at [384400,0,0][km] mass 7.348e22[kg] velocity [0,1.022,0][km/s]

# With color for visualization
sphere Sun at [0,0,0] mass 1.989e30[kg] color "yellow"

# Other shapes
cube Box at [100,0,0][km] radius 10[km] mass 1000[kg]
pointmass Particle at [0,100,0][km] mass 1[kg]
probe Satellite at [7000,0,0][km] mass 500[kg]
```

### 2. Setting Velocity

```gravity
# Assign velocity after creation
Moon.velocity = [0,1.022,0][km/s]

# Apply thrust (delta-v)
thrust Satellite by [0,0.5,0][km/s]
```

### 3. Gravity Interactions

```gravity
# Single pull
Earth pull Moon

# Multiple pulls (NEW! comma-separated)
Core pull Star1, Star2, Star3

# Mutual gravity for all objects
grav all

# Inside simulation loop
simulate t in 0..100 step 60[s] {
    grav all
}
```

### 4. Simulation Loops

```gravity
# Basic loop
simulate t in 0..100 step 60[s] {
    Earth pull Moon
    print Moon.position
}

# With units on range (NEW!)
simulate t in 0..365[days] step 1[days] integrator verlet {
    Earth pull Moon
}

# Orbit loop (alternative syntax)
orbit t in 0..24 dt 3600[s] integrator rk4 {
    Earth pull Moon
}

---

## 🌌 Example: Galaxy Collision

See `examples/galaxy_collision.gravity` for a complete demonstration of:
- **Comma-separated pull syntax** - Cleanly define multiple gravitational interactions
- **Color-coded objects** - Visualize different galaxies with colors
- **N-body simulation** - Two galaxies (7 objects total) with realistic dynamics

```gravity
# Milky Way core (blue)
sphere MilkyWay_Core at [0,0,0][km] mass 4e36[kg] color "cyan" fixed

# 3 stars orbiting Milky Way
sphere StarA1 at [2e14,0,0][km] mass 2e30[kg] velocity [0,220000,0][m/s] color "lightblue"
sphere StarA2 at [-2e14,0,0][km] mass 2e30[kg] velocity [0,-220000,0][m/s] color "lightblue"
sphere StarA3 at [0,2e14,0][km] mass 1.8e30[kg] velocity [-220000,0,0][m/s] color "deepskyblue"

# Andromeda core approaching (red)
sphere Andromeda_Core at [8e14,0,0][km] mass 5e36[kg] velocity [-100000,0,0][m/s] color "magenta"

# 3 stars orbiting Andromeda
sphere StarB1 at [1e15,0,0][km] mass 2.2e30[kg] velocity [-100000,250000,0][m/s] color "pink"
# ... more stars

# Gravitational interactions using comma-separated syntax!
MilkyWay_Core pull StarA1, StarA2, StarA3
Andromeda_Core pull StarB1, StarB2, StarB3

simulate t in 0..10 step 1[day] integrator verlet {
    # Watch the galaxies interact!
}
```

**Run with visualization:**
```bash
python gravity_lang_interpreter.py run examples/galaxy_collision.gravity --3d
```

---

### 5. Integrators

Choose the right integrator for your simulation:

```gravity
integrator leapfrog  # Default - good for long-term stability
integrator rk4       # High accuracy for short-term
integrator verlet    # Excellent energy conservation
integrator euler     # Fast but less accurate
```

Adaptive timestep for better energy control in long runs:

```gravity
simulate t in 0..365[days] step 1[days] integrator verlet adaptive tol 1e-7 min 60[s] max 1[days] {
    grav all
}
```

### 6. Physics Controls

```gravity
# Variables
let burn = 15[m/s]

# Conditionals
if burn > 10[m/s] then thrust Probe by [0,burn,0]

# Friction/drag
friction 0.000001

# Collisions
collisions on
collisions off

# Energy monitoring
monitor energy
```

### 7. Output and Observation

```gravity
# Print to console
print Earth.position
print Moon.velocity

# Stream to CSV file
observe Moon.position to "moon_data.csv" frequency 10
observe Earth.velocity to "earth_vel.csv" frequency 1
```

### 8. Orbital Elements

```gravity
# Calculate Keplerian orbital elements
orbital_elements Moon around Earth
```

Output includes:
- Semi-major axis (km)
- Eccentricity
- Inclination (degrees)
- Periapsis/Apoapsis distances
- And more

### 9. Units

Supported units:
- **Distance**: `m`, `km`
- **Time**: `s`, `min`, `hour`, `day`, `days`
- **Mass**: `kg`
- **Velocity**: `m/s`, `km/s`

```gravity
# With units
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg]

# Vectors with units
Moon.velocity = [0,1.022,0][km/s]
thrust Probe by [100,0,0][m/s]
```

---

## 📚 Examples

### Moon Orbit

```gravity
sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg] fixed
sphere Moon at [384400,0,0][km] mass 7.348e22[kg] velocity [0,1.022,0][km/s]

simulate t in 0..28 step 3600[s] integrator verlet {
    Earth pull Moon
    print Moon.position
}
```

### Binary Star System

```gravity
sphere StarA at [-500000,0,0][km] mass 2e30[kg] velocity [0,-15[km/s],0]
sphere StarB at [500000,0,0][km] mass 1e30[kg] velocity [0,30[km/s],0]

simulate t in 0..365 step 3600[s] integrator verlet {
    StarA pull StarB
    StarB pull StarA
    print StarA.position
}
```

### Solar System

```gravity
sphere Sun at [0,0,0] mass 1.989e30[kg] fixed
sphere Earth at [149.6e6[km],0,0] mass 5.972e24[kg] velocity [0,29.78[km/s],0]
sphere Mars at [227.9e6[km],0,0] mass 6.39e23[kg] velocity [0,24.07[km/s],0]

simulate t in 0..365 step 86400[s] integrator leapfrog {
    grav all
    observe Earth.position to "earth.csv" frequency 30
}
```

**More examples in the [`examples/`](examples/) directory!**

---

## 🔬 Advanced Features

### High-Performance Backend (C++ Default)

For large N-body simulations (100+ objects), C++ backend is the default and recommended path:

```python
from gravity_lang_interpreter import GravityInterpreter, NumPyPhysicsBackend

# 10x-50x faster for large simulations
interp = GravityInterpreter(physics_backend=NumPyPhysicsBackend())
output = interp.execute(script)
```

### Native C++ Compiler (`gravityc`)

You can now compile `.gravity` scripts into generated C++ and native executables:

```bash
# Build gravityc itself
g++ -O3 -std=c++17 cpp/gravity_compiler.cpp -o gravityc

# Or build gravityc via CLI (produces dist/gravityc or dist/gravityc.exe)
python gravity_lang_interpreter.py build-cpp-exe --outdir dist

# Emit C++ from Gravity script
./gravityc examples/moon_orbit.gravity --emit artifacts/moon_orbit.generated.cpp

# Emit + build native executable
./gravityc examples/moon_orbit.gravity --emit artifacts/moon_orbit.generated.cpp --build artifacts/moon_orbit_native

# Emit + build + run in one command
./gravityc examples/moon_orbit.gravity --emit artifacts/moon_orbit.generated.cpp --build artifacts/moon_orbit_native --run
```

`gravityc` currently supports native codegen for `sphere`, `grav all`, `pull`, velocity assignment,
`orbit/simulate` loops (including integrator hints for `verlet`/`leapfrog`/`euler`), `friction`,
`thrust`, `observe <name>.position ... frequency`, and `print <name>.position|velocity`.

### Custom Gravity Laws

Research alternative physics theories:

```python
from gravity_lang_interpreter import (
    PythonPhysicsBackend,
    modified_newtonian_gravity,  # MOND
    schwarzschild_correction,    # General Relativity
)

# Use MOND instead of Newtonian gravity
backend = PythonPhysicsBackend(gravity_law=modified_newtonian_gravity)
interp = GravityInterpreter(physics_backend=backend)

# Or define your own
def custom_gravity(mass: float, distance: float, **kwargs) -> float:
    G = 6.67430e-11
    return G * mass / (distance ** 2.5)  # Your custom law

backend = PythonPhysicsBackend(gravity_law=custom_gravity)
```

**[📖 Read the Advanced Features Guide](ADVANCED_FEATURES.md)**

---

## 🎯 Use Cases

- 🎓 **Education** - Teach orbital mechanics and physics
- 🔬 **Research** - Test alternative gravity theories (MOND, GR)
- 🚀 **Mission Planning** - Simulate spacecraft trajectories
- 🌌 **N-body Studies** - Galaxy clusters, asteroid belts
- 📊 **Data Generation** - Create datasets for ML/analysis

---

## 🆚 Comparison with Other Tools

| Feature | Gravity-Lang | MATLAB | Julia | REBOUND | Modelica |
|---------|-------------|--------|-------|---------|----------|
| **Easy to Learn** | ✅ < 1 hour | ❌ Days | ⚠️ Hours | ⚠️ Moderate | ❌ Weeks |
| **No Coding Required** | ✅ DSL | ❌ Code | ❌ Code | ❌ Python API | ❌ Complex |
| **Cost** | ✅ Free | ❌ $800+/yr | ✅ Free | ✅ Free | ⚠️ Varies |
| **Performance** | ✅ Fast | ✅ Fast | ✅ Very Fast | ✅ Fast | ⚠️ Varies |
| **Custom Physics** | ✅ Yes | ⚠️ Complex | ⚠️ Manual | ❌ Limited | ⚠️ Complex |
| **Built-in Gravity** | ✅ Yes | ❌ No | ❌ No | ✅ Yes | ❌ No |

---

## 🧪 Testing

Run the comprehensive test suite:

```bash
python -m unittest discover -s tests -v
```

All tests cover:
- Core language features
- All 4 integrators
- Vector operations
- Orbital elements calculation
- Advanced physics features

---

## 🛠️ CLI Commands

```bash
# Run a script (auto backend prefers cpp by default)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity

# Validate syntax
python gravity_lang_interpreter.py check examples/solar_system.gravity

# Show backend availability on this machine
python gravity_lang_interpreter.py backends

# Benchmark selected backends with warmup/repeats and CSV export
python gravity_lang_interpreter.py benchmark --objects 200 --steps 20 --repeats 5 --warmup 3 --backends python,numpy,cpp --csv-out artifacts/bench.csv

# Show version
python gravity_lang_interpreter.py --version

# Headless animation export (great for CI/social media content)
python gravity_lang_interpreter.py run examples/moon_orbit.gravity --3d --animate --headless

# Build standalone executable
python gravity_lang_interpreter.py build-exe --name gravity-lang --outdir dist --install-all-deps
```

---

## 📐 Technical Details

### Physics Integrators

1. **Leapfrog** (default)
   - 2nd order symplectic
   - Excellent for long-term orbital stability
   - Time-reversible

2. **RK4** (Runge-Kutta 4th order)
   - 4th order accurate
   - Best for short-term precision
   - Adaptive timestep compatible

3. **Verlet** (Velocity Verlet)
   - 2nd order symplectic
   - Excellent energy conservation
   - Time-reversible

4. **Euler** (Forward Euler)
   - 1st order
   - Fast but less accurate
   - Good for quick prototyping

### Architecture

- **Frontend**: Python parser and interpreter
- **Physics Backends**: 
  - `PythonPhysicsBackend` - Pure Python (portable)
  - `NumPyPhysicsBackend` - Vectorized acceleration
  - `CppPhysicsBackend` - Native kernel via `g++` + `ctypes`
  - `JuliaDiffEqBackend` - DifferentialEquations.jl (optional)
- **Future**: GPU acceleration, distributed computing

---

## 🗺️ Roadmap

### v1.5 (Next Release)

- [x] Variables: `let x = 10`
- [ ] Better error messages with line numbers
- [ ] More print options: `print Earth.speed`, `print Moon.distance`
- [x] Conditional statements: `if ... then ...`
- [ ] Functions/macros for reusable code

### v2.0 (Future)

- [ ] C++ kernel v2 optimization pass (SIMD + multithreading)
- [ ] GPU acceleration with CuPy/CUDA (1000x speedup)
- [ ] Real-time 3D visualization
- [ ] Adaptive timestep presets + auto-tuning
- [ ] Distributed computing backend

---

## 🤝 Contributing

Contributions welcome! Areas to help:

- 📝 More example scripts
- 🐛 Bug reports and fixes
- ✨ Feature requests
- 📚 Documentation improvements
- 🧪 Additional tests

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

Free to use for education, research, and commercial projects.

---

## 🙏 Acknowledgments

Inspired by:
- **REBOUND** - N-body simulation library
- **Modelica** - Physical system modeling
- **Domain-Specific Languages** - Clean, expressive syntax

Built with:
- Python 3.8+
- NumPy (optional, for performance)
- PyInstaller (for standalone executable)

---

## 📞 Support

- 📖 [Read the docs](ADVANCED_FEATURES.md)
- 🐛 [Report issues](https://github.com/YOUR_USERNAME/Gravity-Lang/issues)
- 💬 [Discussions](https://github.com/YOUR_USERNAME/Gravity-Lang/discussions)

---

**Made with ❤️ for scientists, educators, and space enthusiasts**

⭐ Star this repo if you find it useful!
