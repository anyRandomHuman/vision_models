from pathlib import Path
import cv2

# import hydra
import numpy as np

from omegaconf import DictConfig, OmegaConf

# import ultralytics
# from detectors import detic
from detectors.obj_detector import Object_Detector
from detectors.yolo import Yolo_Detrector
import h5py
import time
from tqdm import tqdm


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


def predict_imgs_from_hdf5(
    path: Path, detector: Object_Detector, outpath, task, file="imgs.hdf5"
):
    traj = path.name
    f = h5py.File(path / file, "r")
    out_traj = Path(f"{outpath}/{type(detector)}/{task}/{traj}")
    out_traj.mkdir(parents=True, exist_ok=True)

    total_time = 0
    num_imgs = 0
    for key in f.keys():
        out_cam = out_traj / key
        out_cam.mkdir(exist_ok=True)
        i = 0
        for img_code in tqdm(f[key]):

            img = cv2.imdecode(img_code, 1)

            t = time.time()
            detector.predict(img)
            total_time += time.time() - t
            mask = detector.get_mask_feature()
            union_mask = detector.joint_feature(mask)
            masked_img = detector.get_masked_img(union_mask)
            cv2.imwrite(f"{str(out_cam)}/{i}.jpg", masked_img)

            i += 1

        num_imgs += i
    time_record = out_traj / "time.txt"
    with time_record.open("w") as time_file:
        time_file.write(
            f"totel time: {total_time}\n img num: {num_imgs}\n avg:{total_time/(num_imgs)}"
        )
    f.close()


# @hydra.main(config_path="configs", config_name="pick_placing_config.yaml")
# def main(cfg: DictConfig):
#     detector = hydra.utils.instantiate(cfg.detrctors)
#     path = cfg.path
#     outpath = cfg.outpath
#     task = cfg.task
#     results = ultralytics.YOLO(model="", task="segmentation")


if __name__ == "__main__":

    img = cv2.imread("0.png")

    # obj_det = detic.Detectron(
    #     to_tensor=False,
    #     classes=[
    #         "banana",
    #         "cup",
    #         "carrot",
    #         "saucepan",
    #         "hair_dryer",
    #         "bolt",
    #         "hinge",
    #         "bolt",
    #     ],
    # )

    obj_det = Yolo_Detrector(
        path="/home/alr_admin/david/praktikum/d3il_david/detector_models/yolov8n.pt",
        to_tensor=False,
        device="cuda",
    )
    obj_det.predict(img)
    mask = obj_det.get_mask_feature()
    union_mask = obj_det.joint_feature(mask)
    output = obj_det.get_masked_img(union_mask)
    # output = obj_det.get_masked_img(masked_img)
    # output = obj_det.get_visualized_imgs()
    # mask = np.expand_dims(mask, -1)
    cv2.imwrite("output0.jpg", output)

    # path = Path('/home/i53/student/qwei/alr/data/pickPlacing/2024_08_05-13_22_36')
    # outpath = '/home/i53/student/qwei/alr/prediction_results'
    # detector = detic.Detectron(to_tensor=False)
    # predict_imgs_from_hdf5(path=path, detector=detector, outpath=outpath, task='pickPlacing')
