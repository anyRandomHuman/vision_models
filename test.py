import h5py
import cv2
import glob

# path = "/media/alr_admin/ECB69036B69002EE/Data_less_obs_new_hdf5_downsampled/pickPlacing/2024_08_05-13_22_36/masked_imgs.hdf5"

# f = h5py.File(path, "r")

# ks = f.keys()

# img_codes = f[list(ks)[0]]
# for i, img_code in enumerate(img_codes):
#     img = cv2.imdecode(img_code, 1)
#     cv2.imwrite(f"imgs/{i}.jpg", img)

imgs = glob.glob("imgs/*.jpg")
print(imgs[-2:])
