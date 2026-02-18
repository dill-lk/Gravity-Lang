import os
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

    def test_simulate_loop_and_observe_stream(self):
        out_file = "artifacts/moon_positions.csv"
        if os.path.exists(out_file):
            os.remove(out_file)

        src = """
        sphere Earth at [0,0,0] mass 5.972e24[kg] fixed
        sphere Moon at [384400[km],0,0] mass 7.348e22[kg] velocity [0,1[km],0]
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

    def test_dimension_checked_quantities(self):
        interp = GravityInterpreter()
        g = interp.parse_quantity("6.67430e-11[m^3 kg^-1 s^-2]")
        m = interp.parse_quantity("5.972e24[kg]")
        r = interp.parse_quantity("6371[km]")
        accel = g.mul(m).div(r.pow(2))
        self.assertEqual(accel.dims, {"L": 1, "T": -2})

        with self.assertRaises(ValueError):
            Quantity(1.0, {"L": 1}).add(Quantity(1.0, {"M": 1}))


if __name__ == "__main__":
    unittest.main()
