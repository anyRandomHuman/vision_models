from pathlib import Path
import cv2
# import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf
import ultralytics
from detectors import detic


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


# @hydra.main(config_path="configs", config_name="pick_placing_config.yaml")
# def main(cfg: DictConfig):
#     detector = hydra.utils.instantiate(cfg.detrctors)
#     path = cfg.path
#     outpath = cfg.outpath
#     task = cfg.task
#     results = ultralytics.YOLO(model="", task="segmentation")


if __name__ == "__main__":
    img = cv2.imread("resized38.png")

    obj_det = detic.Detectron(True, obj_classes=[41, 39])

    mask = obj_det.get_feature(img)
    mask = np.expand_dims(mask, -1)
    cv2.imshow("1", np.where(mask, img, np.zeros(img.shape)))
    cv2.waitKey(0)