import unittest
import numpy as np
from analytic_throw_2d import main

class TestAnalyticThrow(unittest.TestCase):
    def test_main(self):
        x0_z0_values = np.linspace([-0.4, 0], [0.4, 0.12], 100)
        xf_zf_values = np.linspace([-0.4, 0], [0.4, 0], 100)
        error_values = np.linspace(0.001, 0.01, 3)

        for x0, z0 in x0_z0_values:
            for xf, zf in xf_zf_values:
                if xf > x0+0.01 and z0-zf > 0.01:
                    for error in error_values:
                        result = main(x0, z0, xf, zf, error)
                        self.assertTrue(result.success, f"Optimization failed for x0={x0}, z0={z0}, xf={xf}, zf={zf}, error={error} error: {result.message}")
                        self.assertTrue(result.success, result)

if __name__ == '__main__':
    unittest.main()