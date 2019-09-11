from util import *
import numpy as np
import matplotlib.pyplot as plt


x_mag = 100
y_mag = 200
z_mag = 20000

_x = [i for i in np.arange(-600., 600., 50)]
_z = [140000 + z_mag * np.cos(i) for i in _x]
_y = np.sin(_x) * 200

# _t = [i for i in np.arange(-1.2, 1.2, 0.01)]
# _x = [i*500 for i in _t]
# _z = [140000 + z_mag * np.cos(i) for i in _t]
# _y = np.sin(_t) * 200

_dy = np.cos(_x)

original_troj = np.vstack([_x, _y, _z]).T
original_troj[:,0] *= x_mag
original_troj[:,1] *= y_mag

noise = np.random.randn(24,3)
noise[:,0] *= x_mag * 50
noise[:,1] *= y_mag * 50
noise[:,2] *= z_mag * 0.1

original_troj = [[i] for i in original_troj]
troj = [[i+j] for i,j in zip(original_troj,noise)]

# print(troj)

# vis(troj,original_troj)
# print(np.array(troj))
tracker = Tracker(dimension=3)
lines,ret = tracker.track(troj)

vis_pred(troj,original_troj,ret)
d1, d2 = compute_diff(troj,original_troj,ret)
print(d1,d2)