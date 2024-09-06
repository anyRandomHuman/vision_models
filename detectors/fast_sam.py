from ultralytics.models.fastsam import FastSAMPredictor
import torch
import numpy as np
import cv2

from detectors.obj_detector import Object_Detector


class FastSAMDetector(Object_Detector):
    def __init__(
        self,
        to_tensor=False,
        device="cuda",
        path="/home/alr_admin/david/praktikum/d3il_david/detector_models/FastSAM-x.pt",
        imgsz=(256, 128),
    ) -> None:
        super().__init__(path=path, to_tensor=to_tensor, device=device)
        overrides = dict(
            task="segment", mode="predict", model=path, save=False, imgsz=imgsz
        )
        self.model = FastSAMPredictor(overrides=overrides)

    def predict(self, img, **kwargs):
        super().predict(img)
        self.prediction = self.model(input)
        self.prediction = self.model.prompt(self.prediction, **kwargs)

    def get_box_feature(self):
        p = self.prediction[0] # type: ignore
        num_boxes = p.boxes.shape[0]

        features = torch.zeros(p.orig_shape + (num_boxes,), dtype=torch.int32)

        b = self.prediction[0].boxes.xyxy.int()

        for i in range(num_boxes):
            box = b[i]
            features[box[1] : box[3], box[0] : box[2], i] = 1

        if not self.to_tensor:
            features = features.cpu().numpy()
        return features

    def get_mask_feature(self):
        p = self.prediction[0]
        return p.masks.data

    def get_bbox(self):
        return self.prediction[0].boxes.xyxy

if __name__ == "__main__":

    # img = cv2.imread(
    #     # "/media/alr_admin/Data/atalay/new_data/pickPlacing/2024_08_05-13_22_36/images/Azure_1/0.png"
    #     "277.png"
    # )
    # img = cv2.resize(img, (128, 256))
    # cv2.imwrite("resized277.jpg", img)

    fsam = FastSAMDetector(to_tensor=False, device="cuda")
    # Prompt inference
    # prompt for cam1
    # bboxes=[[24, 185, 40, 200], [55, 220, 70, 235], [70,170, 95, 205], [40,155, 65, 180]],
    # points=[[85, 200], [50, 170]],
    img = cv2.imread("cam0_1.png")
    fsam.predict(
        img,
        bboxes=[
            [40, 185, 60, 230],
            [85, 185, 120, 225],
            [40, 165, 70, 180],
            [90, 150, 100, 180],
        ],
    )
    # bbox = fsam.get_box_feature()
    # joint_box = fsam.joint_feature(bbox)
    # img = fsam.get_masked_img()

    # cv2.imwrite("test.jpg", img)
    mask = fsam.get_mask_feature()
    joint_mask = fsam.joint_feature(mask)
    
