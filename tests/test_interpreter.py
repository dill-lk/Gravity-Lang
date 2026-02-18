import unittest

from gravity_lang_interpreter import GravityInterpreter


class InterpreterTests(unittest.TestCase):
    def test_object_creation_and_units(self):
        src = "sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg]"
        interp = GravityInterpreter()
        interp.execute(src)
        earth = interp.objects["Earth"]
        self.assertEqual(earth.radius, 6371_000.0)
        self.assertEqual(earth.mass, 5.972e24)

    def test_pull_and_orbit_produces_output(self):
        src = """
        sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg]
        sphere Moon at [384400[km],0,0] radius 1737[km] mass 7.348e22[kg]
        Earth pull Moon
        orbit t in 0..2 dt 3600[s] {
            print Moon.position
        }
        """
        interp = GravityInterpreter()
        output = interp.execute(src)
        self.assertEqual(len(output), 2)
        self.assertTrue(output[0].startswith("Moon.position="))
        self.assertNotEqual(output[0], output[1])

    def test_custom_backend_can_be_injected(self):
        class NoOpBackend:
            def __init__(self):
                self.calls = 0

            def step(self, objects, pull_pairs, dt):
                self.calls += 1

        backend = NoOpBackend()
        src = """
        sphere Earth at [0,0,0] radius 6371[km] mass 5.972e24[kg]
        orbit t in 0..3 dt 1[s] {
            print Earth.position
        }
        """
        interp = GravityInterpreter(physics_backend=backend)
        output = interp.execute(src)
        self.assertEqual(backend.calls, 3)
        self.assertEqual(len(output), 3)


if __name__ == "__main__":
    unittest.main()
