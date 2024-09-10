import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import cv2
import os
from pathlib import Path


class Sam2:
    def __init__(
        self,
        model_cfg="sam2_hiera_t.yaml",
        sam2_checkpoint="models/sam2_hiera_tiny.pt",
        device="cuda",
        to_tensor=False,
    ) -> None:
        self.to_tensor = to_tensor
        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        self.id = 0

    def predict(self, img, **kwargs):
        with torch.inference_mode():
            self.predictor.set_image(img)
            masks, _, _ = self.predictor.predict(**kwargs)
            self.results = masks

    def predict_video(self):

        self.video_segments = (
            {}
        )  # video_segments contains the per-frame segmentation results
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in self.predictor.propagate_in_video(self.state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().squeeze().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def add_boxes(self, boxes, frame_idx):
        for i in range(len(boxes)):
            _, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                box=boxes[i],
                frame_idx=frame_idx,
                obj_id=self.id,
            )
            self.id += 1

    def add_all_points(self, all_points: list, all_labels: np.ndarray, frame_idx):
        for i in range(len(all_points)):
            _, _, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=frame_idx,
                obj_id=self.id,
                points=all_points[i],
                labels=all_labels[i],
            )
            self.id += 1

    def init_states(self, path):
        self.path = path
        self.state = self.predictor.init_state(video_path=path)
        self.id = 0
        return self.state

    def get_feature(self):
        joint_masks = []
        for out_frame_idx in range(len(self.video_segments.keys())):

            joint_mask = np.zeros(
                self.video_segments[out_frame_idx][0].shape, dtype=np.uint8
            )

            for out_obj_id, out_mask in self.video_segments[out_frame_idx].items():
                joint_mask = np.logical_or(joint_mask, out_mask)
            joint_masks.append(joint_mask)
        return joint_masks

    def get_all_masked_imgs(self):
        union_masks = self.get_feature()
        img_paths = sorted(
            Path(self.path).iterdir(), key=lambda p: int(p.name.split('.')[0])
        )
        all_masked = []
        for i, img_path in enumerate(img_paths):
            img = cv2.imread(img_path)
            feature = union_masks[i]
            masked = np.where(
                np.expand_dims(feature, -1).repeat(3, -1),
                img,
                np.zeros(img.shape),
            )
            all_masked.append(masked)
        return all_masked
