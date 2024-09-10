from ultralytics import RTDETR
import torch
import numpy as np
from detectors.obj_detector import Object_Detector


class RTDETR_detector(Object_Detector):
    def __init__(
        self,
        path,
        to_tensor,
        device,
        imgsz=(256, 128),
    ) -> None:
        super().__init__(to_tensor=to_tensor, device=device)

        self.model = RTDETR(path)
        self.imgs = []
        self.imgsz = imgsz

    def predict(self, img):
        super().predict(img)
        self.prediction = self.model.predict(img, imgsz=self.imgsz)

    def track(self, img, **kwargs):
        self.input = img
        self.imgs.append(img)
        self.prediction = self.model.track(
            source=img, persist=True, **kwargs, imgsz=self.imgsz
        )

    def get_mask_feature(self):
        # no mask, return box instead
        return self.get_box_feature()

    def get_box_feature(self):
        # igonre top_n, always retrive all
        p = self.prediction[0]
        num_boxes = p.boxes.shape[0]

        features = np.zeros(p.orig_shape + (num_boxes,), dtype=np.uint8)

        b = self.prediction[0].boxes.xyxy.int()

        for i in range(num_boxes):
            box = b[i]
            features[box[1] : box[3], box[0] : box[2], i] = 1

        if self.to_tensor:
            features = torch.from_numpy(features)
        return features

    def get_Bbox(self):
        box = self.prediction[0].boxes.xyxy.int()
        if self.to_tensor:
            return box
        else:
            return box.cpu().numpy()

    def get_img_with_segment(self):
        return self.prediction[0].plot(
            labels=False, boxes=False, conf=False, probs=False, color_mode="class"
        )


if __name__ == "__main__":

    pass
