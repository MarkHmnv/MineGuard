import cv2
import logging
from datetime import datetime
from ultralytics import YOLO

CONFIDENCE_THRESHOLD = 0.4
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
COLOR = (255, 255, 0)
LINE_THICKNESS = 1
RGB_CAM_INDEX = 'http://192.168.1.100:4747/video'
THERMAL_CAM_INDEX = 1
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILENAME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_mineguard.log'


def initialize_cameras(rgb_cam_index, thermal_cam_index):
    rgb_cam = cv2.VideoCapture(rgb_cam_index)
    thermal_cam = cv2.VideoCapture(thermal_cam_index)
    if not rgb_cam.isOpened() or not thermal_cam.isOpened():
        raise RuntimeError('Error: Unable to open camera.')
    return rgb_cam, thermal_cam


def detect_objects(frame, model):
    return model(frame, conf=CONFIDENCE_THRESHOLD, half=False)


def draw_boxes(image, results, model):
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f'{model.names[cls]} {conf:.2f}'
            logging.info(f'Landmine detected: {label}')
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR, LINE_THICKNESS)
            cv2.putText(image, label, (x1, y1 - 10), FONT, FONT_SCALE, COLOR, LINE_THICKNESS)
    return image


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILENAME)
    ])

    rgb_model = YOLO('rgb_model.pt', task='detect')
    thermal_model = YOLO('thermal_model.pt', task='detect')

    rgb_cam, thermal_cam = initialize_cameras(RGB_CAM_INDEX, THERMAL_CAM_INDEX)

    try:
        while True:
            rgb, rgb_frame = rgb_cam.read()
            thermal, thermal_frame = thermal_cam.read()
            if not rgb or not thermal:
                raise RuntimeError('Error: Unable to read frame.')

            rgb_results = detect_objects(rgb_frame, rgb_model)
            thermal_results = detect_objects(thermal_frame, thermal_model)

            rgb_frame = draw_boxes(rgb_frame, rgb_results, rgb_model)
            thermal_frame = draw_boxes(thermal_frame, thermal_results, thermal_model)

            cv2.imshow('RGB Camera', rgb_frame)
            cv2.imshow('Thermal Camera', thermal_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        rgb_cam.release()
        thermal_cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
