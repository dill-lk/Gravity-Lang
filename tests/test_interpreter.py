import os
import tempfile
import unittest
from unittest.mock import patch

from gravity_lang_interpreter import GravityInterpreter, Quantity, build_executable, main
import unittest

from gravity_lang_interpreter import GravityInterpreter, Quantity


class InterpreterTests(unittest.TestCase):
    def test_object_creation_and_units(self):
        src = "sphere Earth at [0,0,0] mass 5.972e24[kg] radius 6371[km]"
        interp = GravityInterpreter()
        interp.execute(src)
        earth = interp.objects["Earth"]
        self.assertEqual(earth.radius, 6371_000.0)
        self.assertEqual(earth.mass, 5.972e24)

    def test_cube_object_supported(self):
        src = "cube Box at [1,2,3] mass 10[kg]"
        interp = GravityInterpreter()
        interp.execute(src)
        self.assertEqual(interp.objects["Box"].shape, "cube")

    def test_parse_value_allows_whitespace(self):
        interp = GravityInterpreter()
        self.assertEqual(interp.parse_value("  1.5[km]  "), 1500.0)

    def test_vector_with_suffix_unit_supported(self):
        interp = GravityInterpreter()
        self.assertEqual(interp.parse_vector("[1,2,3][km]"), (1000.0, 2000.0, 3000.0))
        self.assertEqual(interp.parse_vector("[0,1,0][km/s]"), (0.0, 1000.0, 0.0))

    def test_unknown_object_in_print_raises_value_error(self):
        interp = GravityInterpreter()
        with self.assertRaises(ValueError):
            interp.execute("print Missing.position")

    def test_simulate_loop_and_observe_stream(self):
        out_file = "artifacts/moon_positions.csv"
        if os.path.exists(out_file):
            os.remove(out_file)

        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
        sphere Moon at [384400,0,0][km] mass 7.348e22[kg] velocity [0,1.022,0][km/s]
        simulate t in 0..3 step 60[s] integrator leapfrog {
            Earth pull Moon
            print Moon.position
            observe Moon.position to "artifacts/moon_positions.csv" frequency 1
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 3)
        self.assertTrue(os.path.exists(out_file))
        with open(out_file, "r", encoding="utf-8") as f:
            rows = [line.strip() for line in f if line.strip()]
        self.assertEqual(rows[0], "step,x,y,z")
        self.assertEqual(len(rows), 4)

    def test_custom_backend_can_be_injected(self):
        class NoOpBackend:
            def __init__(self):
                self.calls = 0
                self.last_integrator = None

            def step(self, objects, pull_pairs, dt, integrator):
                self.calls += 1
                self.last_integrator = integrator

        backend = NoOpBackend()
        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg]
        orbit t in 0..3 dt 1[s] integrator rk4 {
            print Earth.position
        }
        """
        interp = GravityInterpreter(physics_backend=backend)
        output = interp.execute(src)
        self.assertEqual(backend.calls, 3)
        self.assertEqual(backend.last_integrator, "rk4")
        self.assertEqual(len(output), 3)

    def test_rk4_with_tangential_velocity_changes_y(self):
        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
        sphere Moon at [384400000,0,0][m] mass 7.348e22[kg] velocity [0,1022,0][m/s]
        orbit t in 0..2 dt 3600[s] integrator rk4 {
            print Moon.position
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 2)
        self.assertIn(",", output[0])
        self.assertNotIn(", 0.0, 0.0)", output[0])

    def test_velocity_assignment_syntax(self):
        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
        sphere Moon at [384400000,0,0][m] mass 7.348e22[kg]
        Moon.velocity = [0,1022,0][m/s]
        orbit t in 0..2 dt 3600[s] integrator rk4 {
            Earth pull Moon
            print Moon.position
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 2)
        self.assertNotIn(", 0.0, 0.0)", output[0])

    def test_step_physics_statement(self):
        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
        sphere Moon at [384400000,0,0][m] mass 7.348e22[kg]
        Moon.velocity = [0,1022,0][m/s]
        orbit t in 0..2 dt 3600[s] integrator rk4 {
            step_physics(Moon, Earth)
            print Moon.position
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 2)
        self.assertNotIn("384400000", output[0])

    def test_grav_all_applies_mutual_gravity(self):
        src = """
        sphere A at [-1000,0,0][m] mass 1e10[kg]
        sphere B at [1000,0,0][m] mass 1e10[kg]
        grav all
        orbit t in 0..2 dt 1[s] integrator leapfrog {
            print A.position
            print B.position
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 4)
        self.assertNotIn("-1000.0, 0.0, 0.0", output[0])
        self.assertNotIn("1000.0, 0.0, 0.0", output[1])

    def test_friction_reduces_velocity(self):
        src = """
        sphere Probe at [0,0,0] mass 1[kg] velocity [10,0,0][m/s]
        friction 0.1
        orbit t in 0..2 dt 1[s] integrator leapfrog {
            print Probe.velocity
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 2)
        self.assertIn("9.0", output[0])
        self.assertIn("8.1", output[1])

    def test_collision_bounce(self):
        src = """
        sphere A at [-1,0,0] mass 1[kg] radius 1[m] velocity [1,0,0][m/s]
        sphere B at [1,0,0] mass 1[kg] radius 1[m] velocity [-1,0,0][m/s]
        collisions on
        orbit t in 0..2 dt 1[s] integrator leapfrog {
            print A.velocity
            print B.velocity
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 4)
        self.assertIn("A.velocity=(-1.0", output[0])
        self.assertIn("B.velocity=(1.0", output[1])

    def test_dimension_checked_quantities(self):
        interp = GravityInterpreter()
        g = interp.parse_quantity("6.67430e-11[m^3 kg^-1 s^-2]")
        m = interp.parse_quantity("5.972e24[kg]")
        r = interp.parse_quantity("6371[km]")
        accel = g.mul(m).div(r.pow(2))
        self.assertEqual(accel.dims, {"L": 1, "T": -2})

        with self.assertRaises(ValueError):
            Quantity(1.0, {"L": 1}).add(Quantity(1.0, {"M": 1}))

    def test_thrust_statement_changes_velocity(self):
        src = """
        sphere Probe at [0,0,0] mass 1[kg]
        thrust Probe by [0,5,0][m/s]
        print Probe.velocity
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(output[0], "Probe.velocity=(0.0, 5.0, 0.0)")

    def test_monitor_energy_outputs_value(self):
        src = """
        sphere A at [0,0,0] mass 10[kg]
        sphere B at [1,0,0] mass 10[kg]
        A pull B
        monitor energy
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].startswith("system.energy="))

    def test_build_executable_requires_pyinstaller(self):
        with patch("gravity_lang_interpreter.shutil.which", return_value=None):
            with self.assertRaises(RuntimeError):
                build_executable("gravity-test", "dist")

    def test_build_executable_can_auto_install_pyinstaller(self):
        with (
            patch("gravity_lang_interpreter.shutil.which", side_effect=[None, "/usr/bin/pyinstaller"]),
            patch("gravity_lang_interpreter.subprocess.run") as run_mock,
        ):
            build_executable("gravity-test", "dist", auto_install=True)
            self.assertEqual(run_mock.call_count, 2)

    def test_cli_check_command(self):
        with tempfile.NamedTemporaryFile("w", suffix=".gravity", delete=False) as tmp:
            tmp.write("sphere Earth at [0,0,0] mass 5.972e24[kg]\n")
            path = tmp.name

        try:
            with patch("sys.argv", ["gravity_lang_interpreter.py", "check", path]):
                code = main()
            self.assertEqual(code, 0)
        finally:
            os.remove(path)


if __name__ == "__main__":
    unittest.main()
