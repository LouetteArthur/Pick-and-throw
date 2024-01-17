import scipy.optimize
import numpy as np
import time
import argparse
import sys
from pytransform3d.transformations import transform_from_pq, invert_transform
from pytransform3d.rotations import matrix_from_axis_angle, quaternion_from_matrix
import yaml


def compute_optimal_release(x0, z0, xf, zf, max_distance, error):
    eps = 1e-8
    J = 6000

    def objective(x, x0, z0, J)->float:
        distance = np.linalg.norm(x[2:]-[x0, z0])
        return 2*distance/x[0] + x[0]/x[1] + x[1]/J
    
    bounds=[(eps, 10), #speed
            (eps,100), #acceleration
            (x0, max_distance), #x
            (z0, 0.08 + (0.18+0.08)/2)] #z

    def constraint1(x):
        return x[0]/x[1] - x[0]/J

    def constraint2(x):
        distance = np.linalg.norm(x[2:]-[x0, z0])
        return 2*distance/x[0] - x[0]/x[1] - x[1]/J

    def constraint3(x):
        return x[0]*np.cos(theta(x)) * tf(x) + x[2]

    def theta(x):
        return np.arctan((x[3]-z0)/(np.linalg.norm(x[2]-[x0]) + eps))

    def tf(x):
        theta_r = theta(x)
        g=9.81
        term1 = x[0]*np.sin(theta_r)/g
        term2 = max(term1**2 + 2*(x[3]- zf)/g, 0)
        return term1 + np.sqrt(term2)
    
    const1 = scipy.optimize.NonlinearConstraint(constraint1, lb=0, ub=np.inf)
    const2 = scipy.optimize.NonlinearConstraint(constraint2, lb=0, ub=np.inf)
    const3 = scipy.optimize.NonlinearConstraint(constraint3, lb=xf-error, ub=xf+error)

    results = scipy.optimize.minimize(objective, x0=np.array([5, 10, (xf+x0)/2, (zf+z0)/2]), args=(x0, z0, J), bounds=bounds, constraints=(const1, const2, const3))

    return results.x

def transform(p0, pf):
    # Compute the vector from P0 to Pf
    v = np.array([pf[0], pf[1], 0]) - np.array([p0[0], p0[1], 0])
    # Normalize the vector
    v = v / np.linalg.norm(v)
    # Compute the angle between the x-axis and this vector
    angle = np.arccos(np.dot(v, [1, 0, 0]))
    # Adjust the angle based on the y-coordinate of the vector
    if v[1] < 0:
        angle = 2*np.pi - angle
    # Compute the rotation matrix
    R = matrix_from_axis_angle(np.append([0, 0, 1], angle)) #[0, 0, 1] is the axis of rotation
    # Convert the rotation matrix to a quaternion
    q = quaternion_from_matrix(R)
    # Combine the translation vector and quaternion to create a pose vector
    pq = np.hstack((np.array([p0[0], p0[1], 0]), q))
    # Compute the transformation matrix from frame A to frame B
    T_AB = transform_from_pq(pq)
    return T_AB
    
def line_rectangle_intersection(p0, pf, x_min, x_max, y_min, y_max):
    x1, y1 = p0
    x2, y2 = pf
    # Compute the coefficients of the line equation
    m = (y2 - y1) / (x2 - x1)
    c = y1 - m * x1

    # Compute the intersection points with the vertical lines
    x_left = x_min
    y_left = m * x_left + c
    x_right = x_max
    y_right = m * x_right + c

    # Compute the intersection points with the horizontal lines
    y_bottom = y_min
    x_bottom = (y_bottom - c) / m
    y_top = y_max
    x_top = (y_top - c) / m

    # Find the intersection points that are inside the rectangle
    intersections = []
    if x_left >= x_min and x_left <= x_max and y_left >= y_min and y_left <= y_max:
        intersections.append((x_left, y_left))
    if x_right >= x_min and x_right <= x_max and y_right >= y_min and y_right <= y_max:
        intersections.append((x_right, y_right))
    if x_bottom >= x_min and x_bottom <= x_max and y_bottom >= y_min and y_bottom <= y_max:
        intersections.append((x_bottom, y_bottom))
    if x_top >= x_min and x_top <= x_max and y_top >= y_min and y_top <= y_max:
        intersections.append((x_top, y_top))

    return intersections

def read_workspace():
        # read the boundaries of the workspace in the config file (yaml)
        with open("config/workspace.yaml", 'r') as workspace_file:
            workspace = yaml.safe_load(workspace_file)

        workspace = workspace['workspace']
        return workspace

def main (p0, pf, error):
    workspace = read_workspace()
    intersections = line_rectangle_intersection(p0[:2], pf[:2], workspace['x_min'], workspace['x_max'], workspace['y_min'], workspace['y_max'])
    if len(intersections) < 2:
        print("Less than 2 intersections")
        sys.exit()
    for i in intersections:
        if i[1] > p0[1]:
            max_distance = np.linalg.norm(np.array(i) - p0[:2])

    T = transform(p0, pf)
    T_inv = invert_transform(T)
    p0_b = np.dot(T_inv, np.append(p0, 1))
    pf_b = np.dot(T_inv, np.append(pf, 1))
    x0= p0_b[0]
    z0 = p0_b[2]

    xf= pf_b[0]
    zf = pf_b[2]
    v, a, xr, zr = compute_optimal_release(x0, z0, xf, zf, max_distance, error)
    pr = np.array([xr, 0, zr])
    T_inv = invert_transform(T)
    pr_in_robot_frame = np.dot(T, np.append(pr, 1))
    return np.append(pr_in_robot_frame[:3], v).tolist()
    
if __name__ == "__main__":
    p0 = np.array([-0.2, 0.2, 0.085])
    pf = np.array([0.6, 0.7, 0.085])
    error = 0.01

    print(main(p0, pf, error))

