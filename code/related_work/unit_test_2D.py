import unittest
import numpy as np
from time_opt_PaT import main, read_workspace

class TestAnalyticThrow(unittest.TestCase):
    def test_main(self):
        workspace = read_workspace()
        x_min = workspace['x_min']
        x_max = workspace['x_max']
        y_min = workspace['y_min']
        y_max = workspace['y_max']
        z = 0.08
        p0_values = np.linspace([x_min, y_min, 0.08], [x_max, y_max, 0.08], 100)
        pf_values = np.linspace([x_max, y_max, 0.08], [x_max+0.2, y_max+0.5, 0.08], 100)
        error_values = np.linspace(0.001, 0.01, 3)


        for p0 in p0_values:
            for pf in pf_values:
                for error in error_values:
                    if p0[0] == pf[0] and p0[1] == pf[1]:
                        continue
                    result = main(p0, pf, error)
                    self.assertTrue(self.is_in_workspace(result))

    def is_in_workspace(self, p):
        workspace = read_workspace()
        x_min = workspace['x_min']
        x_max = workspace['x_max']
        y_min = workspace['y_min']
        y_max = workspace['y_max']
        z_min = workspace['z_min']
        z_max = workspace['z_max']
        if p[0] >= x_min and p[0] <= x_max and p[1] >= y_min and p[1] <= y_max and p[2] >= z_min and p[2] <= z_max and p[3] >= 0 and p[3] <= 10:
            return True
        else:
            return False
                    
if __name__ == '__main__':
    unittest.main()