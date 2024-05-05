import os
import time as tm
from datetime import datetime
import platform

import cv2
import folium
import numpy as np
import sounddevice as sd
import torch
from ultralytics import YOLO
from torchaudio.transforms import MelSpectrogram

from base_log import log, DATE_FORMAT
from gps import GPSModule
from landmine import Landmine
from model.constants import DEVICE, class_mapping, SAMPLE_RATE, NUM_CLASSES, NUM_SAMPLES
from model.model import DualPathNet

DEBUG = False
IS_WINDOWS = platform.system() == 'Windows'
AUDIO_VOLUME_THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.4
MIN_TIME_DIFFERENCE = 0.2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.4
COLOR = (255, 255, 0)
LINE_THICKNESS = 1
RGB_CAM_INDEX = 0
THERMAL_CAM_INDEX = 1
SAVE_PATH = 'output'


# Global variables
recording = False
audio_data = []
recording_st = None
rgb_frame = None
thermal_frame = None


def initialize_cameras(rgb_cam_index, thermal_cam_index) -> tuple:
    rgb_cam = cv2.VideoCapture(rgb_cam_index)
    if IS_WINDOWS:
        thermal_cam = cv2.VideoCapture(thermal_cam_index)
    else:
        thermal_cam = cv2.VideoCapture(f'/dev/video{thermal_cam_index}', cv2.CAP_V4L2)
    thermal_cam.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
    if not rgb_cam.isOpened() or not thermal_cam.isOpened():
        raise RuntimeError('Error: Unable to open camera.')
    return rgb_cam, thermal_cam


def get_heatmap(frame: np.ndarray) -> np.ndarray:
    if IS_WINDOWS:
        frame = np.reshape(frame[0], (384, 256, 2))
    imdata, _ = np.array_split(frame, 2)
    imdata = cv2.cvtColor(imdata, cv2.COLOR_YUV2BGR_YUYV)
    imdata = cv2.convertScaleAbs(imdata)
    return cv2.applyColorMap(imdata, cv2.COLORMAP_BONE)


def detect_objects(frame: np.ndarray, model: YOLO, conf: float = 0.4) -> list:
    return model(frame, conf=conf, half=False, verbose=DEBUG)


def get_yolo_classes(image: np.ndarray, results: list) -> set:
    found = set()
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            conf = box.conf[0]
            label = f'{result.names[cls]} {conf:.2f}'
            found.add(result.names[cls])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), COLOR, LINE_THICKNESS)
            cv2.putText(image, label, (x1, y1 - 10), FONT, FONT_SCALE, COLOR, LINE_THICKNESS)
    return found


def cut_if_necessary(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    if waveform.shape[1] > num_samples:
        waveform = waveform[:, :num_samples]
    return waveform


def right_pad_if_necessary(waveform: torch.Tensor, num_samples: int) -> torch.Tensor:
    length = num_samples - waveform.shape[1]
    if length > 0:
        waveform = torch.nn.functional.pad(waveform, (0, length))
    return waveform


def load_mel_spec(waveform: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(waveform).to(DEVICE)
    waveform = torch.transpose(waveform, 0, 1)
    waveform = cut_if_necessary(waveform, NUM_SAMPLES)
    waveform = right_pad_if_necessary(waveform, NUM_SAMPLES)
    waveform = mel_spec(waveform)
    return waveform.unsqueeze(0)


def classify_metal(waveform: torch.Tensor) -> str:
    with torch.no_grad():
        predictions = metal_detector_model(waveform)
        predicted_index = predictions.argmax(dim=1)
        predicted = class_mapping[predicted_index]
    return predicted


def create_landmine(
        detected_at: float = None,
        coordinates: tuple = None,
        rgb_classes: set = None,
        thermal_classes: set = None,
        metal_detector_class: str = None
) -> Landmine:
    detected_at = detected_at or tm.time()
    lat, lon = coordinates or gps.get_gps_position()
    detected_at_str = datetime.fromtimestamp(detected_at).strftime(DATE_FORMAT)
    rgb_path = os.path.join(SAVE_PATH, f'rgb_{detected_at_str}.png')
    thermal_path = os.path.join(SAVE_PATH, f'thermal_{detected_at_str}.png')
    cv2.imwrite(rgb_path, rgb_frame)
    cv2.imwrite(thermal_path, thermal_frame)

    all_classes = set()
    all_classes.update(rgb_classes or set())
    all_classes.update(thermal_classes or set())
    if metal_detector_class:
        all_classes.add(metal_detector_class)

    landmine = Landmine(
        classes=all_classes,
        rgb_image=rgb_path,
        thermal_image=thermal_path,
        latitude=lat,
        longitude=lon,
        detected_at=detected_at,
        num_sensors_detected=bool(rgb_classes) + bool(thermal_classes) + bool(metal_detector_class)
    )
    log.info(f'Detected landmine: {landmine}')
    return landmine


def record_metal_and_classify(indata, frames, time, status):
    global recording, audio_data, recording_st

    amplitude = np.max(np.abs(indata))
    if amplitude > AUDIO_VOLUME_THRESHOLD or recording:
        if not recording:
            recording = True
            recording_st = tm.time()
        audio_data.append(indata.copy())
    if recording and tm.time() - recording_st >= 1:
        recording = False
        coordinates = gps.get_gps_position()
        waveform = np.concatenate(audio_data, axis=0)
        audio_data.clear()
        waveform = load_mel_spec(waveform)
        predicted = classify_metal(waveform)

        if predicted == 'other':
            return

        # Finds the nearest mine in the list of landmines by timestamp detected_at
        # Pre-screens out all mines with a difference in timestamp greater than MIN_TIME_DIFFERENCE
        detected_at = recording_st
        nearest_landmine = min(
            (mine for mine in landmines if abs(mine.detected_at - detected_at) < MIN_TIME_DIFFERENCE),
            key=lambda mine: abs(mine.detected_at - detected_at),
            default=None
        )

        if nearest_landmine is not None:
            nearest_landmine.classes.add(predicted)
            nearest_landmine.num_sensors_detected += 1
        else:
            landmine = create_landmine(
                detected_at=detected_at,
                coordinates=coordinates,
                metal_detector_class=predicted
            )
            landmines.append(landmine)

        save_map(landmines)


def save_map(landmines: list[Landmine]):
    if not landmines:
        return
    data = np.array([landmine.get_location() for landmine in landmines])
    m = folium.Map(location=np.mean(data, axis=0).tolist())
    for landmine in landmines:
        popup = folium.Popup(landmine.to_popup_html())
        folium.Marker(
            landmine.get_location(),
            popup=popup,
            icon=folium.Icon(color='red' if landmine.num_sensors_detected > 1 else 'orange')
        ).add_to(m)

    sw = np.min(data, axis=0).tolist()
    ne = np.max(data, axis=0).tolist()
    m.fit_bounds([sw, ne])

    m.save('map.html')


def main():
    global rgb_frame, thermal_frame
    try:
        with sd.InputStream(
                device=0, callback=record_metal_and_classify, channels=1, samplerate=SAMPLE_RATE
        ):
            while True:
                rgb, rgb_frame = rgb_cam.read()
                thermal, thermal_frame = thermal_cam.read()
                thermal_frame = get_heatmap(thermal_frame)
                if not rgb or not thermal:
                    raise RuntimeError('Error: Unable to read frame.')

                rgb_results = detect_objects(rgb_frame, rgb_model, conf=0.4)
                thermal_results = detect_objects(thermal_frame, thermal_model, conf=0.1)
                rgb_classes = get_yolo_classes(rgb_frame, rgb_results)
                thermal_classes = get_yolo_classes(thermal_frame, thermal_results)

                if rgb_classes or thermal_classes:
                    landmine = create_landmine(
                        detected_at=tm.time(),
                        coordinates=gps.get_gps_position(),
                        rgb_classes=rgb_classes,
                        thermal_classes=thermal_classes
                    )
                    landmines.append(landmine)
                    save_map(landmines)

                if DEBUG:
                    cv2.imshow('RGB Camera', rgb_frame)
                    cv2.imshow('Thermal Camera', thermal_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        log.info('Shutting down...')
        rgb_cam.release()
        thermal_cam.release()
        cv2.destroyAllWindows()
        gps.power_down()


if __name__ == '__main__':
    log.info('Initializing...')

    start_time = tm.time()
    os.makedirs(SAVE_PATH, exist_ok=True)

    gps = GPSModule()

    rgb_cam, thermal_cam = initialize_cameras(RGB_CAM_INDEX, THERMAL_CAM_INDEX)
    rgb_model = YOLO('rgb_model.pt', task='detect')
    thermal_model = YOLO('thermal_model.pt', task='detect')

    mel_spec = MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64).to(DEVICE)
    metal_detector_model = DualPathNet(num_classes=NUM_CLASSES).to(DEVICE)
    metal_detector_model.load_state_dict(torch.load('best.pt'))
    metal_detector_model = torch.jit.optimize_for_inference(torch.jit.script(metal_detector_model.eval()))

    landmines: list[Landmine] = []

    log.info(f'Initialized in {tm.time() - start_time:.2f} seconds')
    main()
