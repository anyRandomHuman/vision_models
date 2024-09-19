import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor
import cv2
import os
from pathlib import Path
from collections import OrderedDict
from detectors.obj_detector import Object_Detector


class Sam2(Object_Detector):
    def __init__(
        self,
        model_cfg="sam2_hiera_t.yaml",
        sam2_checkpoint="models/sam2_hiera_tiny.pt",
        device="cuda",
        to_tensor=False,
        inference=False,
        use_n_frame=1,
        imgsz=(256, 128),
    ) -> None:
        self.to_tensor = to_tensor
        self.device = device
        self.predictor = build_sam2_video_predictor(
            model_cfg, sam2_checkpoint, device=device
        )
        self.id = 0
        self.imgsz = imgsz
        self.inference = inference
        self.use_n_frame = use_n_frame  # not yet used
        self.prediction = None

    def predict(self, input, **kwargs):
        super().predict(input)
        with torch.inference_mode():
            if self.inference and self.prediction:
                self.predict_inference(input, **kwargs)
            else:
                self.predictor.set_image(input)
                masks, _, _ = self.predictor.predict(**kwargs)
                self.prediction = masks

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

    def predict_inference(self, input, **kwargs):
        with torch.inference_mode():
            self.video_segments = {}
            self.init_state([self.input, input], **kwargs)
            self.predict_video()

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

    def init_state_list(self, imgls):
        self.state = self.init_state(imgls=imgls)
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
            Path(self.path).iterdir(), key=lambda p: int(p.name.split(".")[0])
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

    def get_box_feature(self):
        pass

    def get_mask_feature(self):
        pass

    @torch.inference_mode()
    def init_state(
        self,
        imgls,
        offload_video_to_cpu=False,
        offload_state_to_cpu=False,
    ):
        """Initialize an inference state."""
        compute_device = self.device  # device of the model
        images, video_height, video_width = load_video_frames_from_list(
            imgls=imgls,
            image_size=self.imgsz,
            offload_video_to_cpu=offload_video_to_cpu,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = images
        inference_state["num_frames"] = len(images)
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        return inference_state


def load_video_frames_from_list(
    imgls,
    image_size,
    offload_video_to_cpu=False,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device=torch.device("cuda"),
):
    """
    Load the video frames from a directory of JPEG files ("<frame_index>.jpg" format).

    The frames are resized to image_size x image_size and are loaded to GPU if
    `offload_video_to_cpu` is `False` and to CPU if `offload_video_to_cpu` is `True`.

    You can load a frame asynchronously by setting `async_loading_frames` to `True`.
    """

    num_frames = len(imgs)
    if num_frames == 0:
        raise RuntimeError(f"no images")
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    images = torch.zeros(num_frames, 3, image_size, image_size, dtype=torch.float32)
    for n, img_path in enumerate(imgs):
        images[n], video_height, video_width = _load_img_as_tensor(img_path)
    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)
    # normalize by mean and std
    images -= img_mean
    images /= img_std
    return images, video_height, video_width


def _load_img_as_tensor(img_np):
    img_np = img_np / 255.0
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    return img, img.shape[0], img.shape[1]
