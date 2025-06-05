import numpy as np

inner = np.load('times_inner_5.npy')
outer = np.load('times_outer_5.npy')

interval = np.concatenate((inner, outer), axis=0)

np.savetxt('intervals_5.txt', interval, fmt='%f')