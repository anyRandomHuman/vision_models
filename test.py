import h5py

path = '/home/i53/student/qwei/alr/data/pickPlacing/2024_08_05-13_22_36/imgs.hdf5'

f = h5py.File(path, 'r')

ks = f.keys()
print(ks)
print(len(f[list(ks)[0]]))