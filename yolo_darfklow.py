from darkflow.net.build import TFNet

from object_detector import ObjectDetector


class YOLODarkflowDetector(ObjectDetector):
    def __init__(self, cfg_path, weights_path):
        options = {"model": cfg_path,
                   "load": weights_path, "threshold": 0.01}

        self.tfnet = TFNet(options)

    def detect(self, frame, threshold=0.1):
        results = self.tfnet(frame)
        return self.__boxes_coordinates(results. threshold)

    def __boxes_coordinates(self, results, threshold):
        boxes = []
        for i in results:
            if i['confidence'] <= threshold: continue
            boxes.append([
                (i['topleft']['x'], i['topleft']['y']),
                (i['bottomright']['x'], i['bottomright']['y']),
                i['confidence'],
                i['label']
            ])

        return boxes