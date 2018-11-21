from abc import ABC, abstractmethod


class ObjectDetector(ABC):
    @abstractmethod
    def detect(self, frame, threshold=0.0):
        pass