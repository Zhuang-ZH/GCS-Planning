import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
from calc_splines import calc_splines
import numpy as np

# --- PARAMETERS ---
CLOSED = True

# --- IMPORT TRACK ---
# load data from csv file
csv_data_temp = np.loadtxt(os.path.dirname(__file__) + '/output.csv',
                            comments='#', delimiter=',')

# get coords and track widths out of array
reftrack = csv_data_temp[:, 0:4]
print(reftrack)
psi_s = None
psi_e = None

# --- CALCULATE MIN CURV ---
if CLOSED:
    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=np.vstack((reftrack[:, 0:2], reftrack[0, 0:2])))
else:
    reftrack = reftrack[200:600, :]
    coeffs_x, coeffs_y, M, normvec_norm = calc_splines(path=reftrack[:, 0:2],
                                                        psi_s=psi_s,
                                                        psi_e=psi_e)

    # extend norm-vec to same size of ref track (quick fix for testing only)
    normvec_norm = np.vstack((normvec_norm[0, :], normvec_norm))

# --- PLOT RESULTS ---
bound1 = reftrack[:, 0:2] - normvec_norm * np.expand_dims(reftrack[:, 2], axis=1)
bound2 = reftrack[:, 0:2] + normvec_norm * np.expand_dims(reftrack[:, 3], axis=1)
print(bound1)
print(bound2)

plt.plot(reftrack[:, 0], reftrack[:, 1], ":")
plt.plot(bound1[:, 0], bound1[:, 1], 'k')
plt.plot(bound2[:, 0], bound2[:, 1], 'k')
plt.axis('equal')
plt.show()