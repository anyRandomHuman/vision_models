from pathlib import Path
import cv2
# import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf
# import ultralytics
from detectors import detic
from detectors.obj_detector import Object_Detector
import h5py
import time

def segment_all_imgs(path: Path, detector, outpath, task):
    traj = path.name
    for img_path in path.iterdir():
        img = cv2.imread(img_path.name)
        detector.predict()
        masked_img = detector.get_masked_img()
        cv2.imwrite(
            f"{outpath}/{type(detector)}/{task}/{traj}",
            masked_img,
        )

def segment_imgs_from_hdf5(path: Path, detector:Object_Detector, outpath, task, file='imgs.hdf5'):
    traj = path.name
    f = h5py.File(path / file, 'r')
    out_traj = Path(f"{outpath}/{type(detector)}/{task}/{traj}")
    out_traj.mkdir(parents=True)
    
    total_time = 0
    for key in f.keys():
        out_cam = out_traj / key
        out_cam.mkdir()
        i = 0
        for img_code in f[key]:
            img = cv2.imdecode(img_code, 1)
            
            t = time.time()
            detector.predict(img)
            total_time += time.time() - t
            mask = obj_det.get_mask_feature()
            union_mask = obj_det.joint_feature(mask)
            masked_img = obj_det.get_masked_img(union_mask)
            cv2.imwrite(f'{str(out_cam)}/{i}.jpg', masked_img)
            
    time_record = out_traj / 'time.txt'
    with time_record.open('w') as f:
        f.write(f'totel time: {total_time}\n img num: {2*i}\n avg:{total_time/(2*i)}')
    f.close()

# @hydra.main(config_path="configs", config_name="pick_placing_config.yaml")
# def main(cfg: DictConfig):
#     detector = hydra.utils.instantiate(cfg.detrctors)
#     path = cfg.path
#     outpath = cfg.outpath
#     task = cfg.task
#     results = ultralytics.YOLO(model="", task="segmentation")


if __name__ == "__main__":
    
    img = cv2.imread("resized38.png")

    obj_det = detic.Detectron(to_tensor=False)
    
    obj_det.predict(img)
    mask = obj_det.get_mask_feature()
    union_mask = obj_det.joint_feature(mask)
    masked_img = obj_det.get_masked_img(union_mask)
    # mask = np.expand_dims(mask, -1)
    cv2.imwrite("output.jpg", masked_img)
    # cv2.waitKey(0)