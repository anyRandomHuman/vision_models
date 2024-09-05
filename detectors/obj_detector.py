from ultralytics import YOLO
from abc import ABC, abstractmethod


class Object_Detector(ABC):
    @abstractmethod
    def __init__(self, path) -> None: ...

    @abstractmethod
    def predict(self, img): ...

    @abstractmethod
    def get_Bbox(self): ...

    @abstractmethod
    def get_mask_feature(self): ...

    @abstractmethod
    def joint_feature(self, featrue): ...

    @abstractmethod
    def get_masked_img(self, feature): ...
