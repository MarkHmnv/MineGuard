from dataclasses import dataclass
from datetime import datetime
from typing import Union


@dataclass
class Landmine:
    classes: set
    rgb_image: Union[str, None]
    thermal_image: Union[str, None]
    latitude: float
    longitude: float
    detected_at: float
    num_sensors_detected: int

    def to_popup_html(self) -> str:
        """Returns a popup HTML with the landmine details for the folium map"""
        rgb_image = self.rgb_image.replace('\\', '/')
        thermal_image = self.thermal_image.replace('\\', '/')
        detected_at = datetime.fromtimestamp(self.detected_at).strftime('%Y-%m-%d %H:%M:%S')
        return (f'Detected at: {detected_at} <br> Landmine type: {self.classes} <br> '
                f'Location: {self.latitude:.6f}, {self.longitude:.6f} <br> '
                f'<img src="{rgb_image}" width="200"> <img src="{thermal_image}" width="200">')

    def get_location(self) -> tuple:
        """Returns a tuple with the landmine latitude and longitude"""
        return self.latitude, self.longitude

    def __repr__(self):
        return f'Type: {self.classes}, Location: {self.latitude:.6f}, {self.longitude:.6f}'