import unittest

from tests.haner.gate_tests.cccx import Test


class CCCXTestCase(unittest.TestCase):
    def test_triple_controlled_not(self):
        test = Test(self)

        params = [{'ctrl': ctrl, 'x': x, 'g': g}
                  for g in range(2) for x in range(2) for ctrl in reversed(range(8))]
        test.run_subtests(params)
