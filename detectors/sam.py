import torch
import numpy as np
import cv2
from ultralytics.models.sam import Predictor as SAMPredictor

from detectors.obj_detector import Object_Detector


class SAMDetector(Object_Detector):
    def __init__(
        self,
        to_tensor=False,
        device="cuda",
        path="/home/alr_admin/david/praktikum/d3il_david/detector_models/sam_b.pt",
        imgsz=(256, 128),
        track=False,
    ) -> None:
        super().__init__(to_tensor=to_tensor, device=device)
        overrides = dict(
            task="segment", mode="predict", model=path, save=False, imgsz=imgsz
        )
        self.model = SAMPredictor(overrides=overrides)

        self.track = track
        self.prediction = None

    def predict(self, img, **kwargs):
        super().predict(img)
        self.model.set_image(img)
        if self.track and self.prediction:
            last_bbox = self.prediction[0].boxes.xywh
            boxes = []
            for box in last_bbox:
                x = int(box[0] + box[2] / 4)
                x1 = int(box[0] + 3 * box[2] / 4)
                y = int(box[1] + box[3] / 4)
                y1 = int(box[1] + 3 * box[3] / 4)
                boxes.append([x, x1, y, y1])
            kwargs["bboxes"] = boxes
        self.prediction = self.model(**kwargs)

    def get_box_feature(self):
        p = self.prediction[0]  # type: ignore
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
        p = self.prediction[0].masks.data.to(self.device).permute((1, 2, 0))
        if not self.to_tensor:
            p = p.cpu().numpy()
        return p

    def get_bbox(self):
        return self.prediction[0].boxes.xyxy
