from pathlib import Path
import cv2
import hydra
import numpy as np

import wandb
from omegaconf import DictConfig, OmegaConf


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


@hydra.main(config_path="configs", config_name="pick_placing_config.yaml")
def main(cfg: DictConfig):
    detector = hydra.utils.instantiate(cfg.detrctors)
    path = cfg.path
    outpath = cfg.outpath
    task = cfg.task
    segment_all_imgs(path, detector, outpath, task)
