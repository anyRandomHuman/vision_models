from pathlib import Path
import cv2

# import ultralytics
from detectors.obj_detector import Object_Detector

from detectors.detic import Detectron

# from detectors.yolo import Yolo_Detrector
# from detectors.rtdetr import RTDETR_detector
# from detectors.sam import SAMDetector
# from detectors.fast_sam import FastSAMDetector
# from detectors.sam_detector import Sam

# from detectors.sam2 import Sam2
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
    from pathlib import Path
    import h5py
    import cv2
    import torch

    # detector = Yolo_Detrector("models/yolov8n.pt", False, "cuda")
    # detector = RTDETR_detector(
    #     path="models/rtdetr-l.pt", to_tensor=False, device="cuda"
    # )

    # detector.model.set_classes(["pan", "bowl", "banana", "carrot"])
    #
    # classes=['banana','bowl', 'toy','cup','saucepan','carrot']
    detector = Detectron(
        to_tensor=False,
        device="cuda",
        to_detect=["banana", "bowl", "toy", "cup", "saucepan", "measuring_cup"],
    )
    # detector = FastSAMDetector(to_tensor=False, device="cuda", track=True)
    # from detectors.ultralytics_sam2 import Sam2_Detector

    # detector = Sam2_Detector(imgsz=(256, 128))

    # detector = Sam()
    # detector = SAMDetector()
    f = h5py.File("/home/alr_admin/david/vision_models/imgs.hdf5", "r")
    k = list(f.keys())[0]
    for i, imgcode in enumerate(f[k]):
        # path = Path(
        #     "/home/i53/student/qwei/alr/data/pickPlacing/2024_08_05-13_22_36/imgs.hdf5"
        # )
        # img_paths = sorted(Path(path).iterdir(), key=lambda p: int(p.name.split(".")[0]))
        # for i, img_path in enumerate(img_paths[:10]):
        img = cv2.imdecode(imgcode, 1)
        # img = cv2.imread(str(img_path))
        # detector.track(img)
        detector.predict(img)
        #     detector.predict(
        #         img,
        #         # bboxes=[[32, 166, 96, 206]],
        #         bboxes=[
        #             [40, 185, 60, 230],
        #             [85, 185, 120, 225],
        #             [40, 165, 70, 180],
        #             [90, 150, 100, 180],
        #         ],
        #         # texts="a photo of cups",
        #     )

        feature = detector.get_mask_feature()
        uf = detector.joint_feature(feature)
        result = detector.get_masked_img(uf)
        cv2.imwrite(f"imgs/{i}.jpg", result)
    # print(detector.detected_classes)
    # import os

    # outpath = path.parent / "test"

    # for img_path in path.iterdir():
    #     img = cv2.imread(str(img_path))
    #     img = cv2.resize(img, (128, 256))
    #     cv2.imwrite(str(outpath / f'{img_path.name.split(".")[0]}.jpg'), img)

    # from detectors.sam2 import Sam2
    # from detectors.ultralytics_sam2 import Sam2_Detector

    # detector = Sam2(
    #     to_tensor=False,
    #     device="cuda",
    # )
    # detector = Sam2_Detector(inference=True)
    # f = h5py.File("/home/alr_admin/david/vision_models/imgs.hdf5", "r")
    # k = list(f.keys())[0]

    # for i, img_code in enumerate(f[k]):
    #     img = cv2.imdecode(img_code, 1)
    #     cv2.imwrite(f"orig_imgs/{i}.jpg", img)

    # from glob import glob

    # def img_file_key(p: Path):
    #     return int(p.name.partition(".")[0])

    # imgs_paths = sorted(glob("orig_imgs/*"), key=lambda x: int(x.split('/')[-1].split('.')[0]))
    # img_path = imgs_paths[0]
    # detector.predict(
    #     img_path,
    #     bboxes=[
    #         [40, 185, 60, 230],
    #         [85, 185, 120, 225],
    #         [40, 165, 70, 180],
    #         [90, 150, 100, 180],
    #     ],
    # )
    # for i in range(1, len(imgs_paths),1):
    #     detector.predict(imgs_paths[i - 1 : i + 1])
    #     results = detector.get_mask_feature()
    #     union = detector.joint_feature(results)
    #     masked = detector.get_masked_img(union)
    #     cv2.imwrite(f"imgs/{i}.jpg", masked)
