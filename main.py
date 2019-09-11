from util import *
import numpy as np
import matplotlib.pyplot as plt


x_mag = 100
y_mag = 200
z_mag = 20000

# _x = [i for i in np.arange(-600., 600., 50)]
# _z = [140000 + z_mag * np.cos(i) for i in _x]
# _y = np.sin(_x) * 200

_t = [i for i in np.arange(-6, 6, 0.03)]

_x0 = [i * 500 for i in _t]
_z0 = [140000 + z_mag * np.cos(i) for i in _t]
_y0 = np.sin(_t) * 200

_x1 = [i * 500 for i in _t]
_z1 = [140000 + z_mag * np.sin(i) for i in _t]
_y1 = np.cos(_t) * 200

# _dy = np.cos(_x)

original_troj0 = np.vstack([_x0, _y0, _z0]).T
original_troj0[:,0] *= x_mag
original_troj0[:,1] *= y_mag

noise0 = np.random.randn(*(np.array(original_troj0).shape))
noise0[:,0] *= x_mag * 50
noise0[:,1] *= y_mag * 50
noise0[:,2] *= z_mag * 0.1

original_troj1 = np.vstack([_x1, _y1, _z1]).T
original_troj1[:,0] *= x_mag
original_troj1[:,1] *= y_mag

noise1 = np.random.randn(*(np.array(original_troj1).shape))
noise1[:,0] *= x_mag * 50
noise1[:,1] *= y_mag * 50
noise1[:,2] *= z_mag * 0.1

original_troj = [[original_troj0[i], original_troj1[i]] for i in range(len(original_troj0))]
troj = [[original_troj0[i] + noise0[i], original_troj1[i] + noise1[i]] for i in range(len(original_troj0))]

# print(troj)

# vis(troj,original_troj)
# print(np.array(troj))
tracker = Tracker(dimension=3,Q=0.001,R=10)
lines,ret = tracker.track(troj)
d1, d2 = compute_diff(troj,original_troj,ret)
print(d1,d2)
vis_pred2(troj,original_troj,ret)
