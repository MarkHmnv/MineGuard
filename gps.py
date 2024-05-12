from typing import Union

import Jetson.GPIO as GPIO
import serial
import time

from base_log import log


class GPSModule:
    """Class for controlling SIM7600X GPS module"""
    def __init__(self, power_key_pin=6):
        self.power_key_pin = power_key_pin
        self.ser = serial.Serial('/dev/ttyTHS1', 115200)
        self.ser.reset_input_buffer()
        self._check_start()
        # Need to call a function to set up the GPS for later use
        self.get_gps_position(initialize=True)

    def get_gps_position(self, initialize=False) -> tuple:
        response = None

        while response is None:
            response = self._send_at_command('AT+CGPSINFO', '+CGPSINFO: ')
            if initialize:
                time.sleep(1)

        return self._convert_coordinates_to_gps(response)

    def power_down(self):
        log.info('Powering down GPS...')
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.power_key_pin, GPIO.OUT)
        GPIO.output(self.power_key_pin, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.power_key_pin, GPIO.LOW)
        time.sleep(18)
        self.ser.close()
        GPIO.cleanup()
        log.info('GPS is powered down')

    def _check_start(self):
        while True:
            self.ser.write('AT\r\n'.encode())
            time.sleep(0.1)
            if self.ser.in_waiting:
                time.sleep(0.01)
                response = self.ser.read(self.ser.in_waiting).decode()
                log.info('Attempting to start GPS...')
                if 'OK' in response:
                    log.info('GPS is ready')
                    return
            else:
                self._power_on()

    def _power_on(self):
        log.info('Starting GPS...')
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.power_key_pin, GPIO.OUT)
        time.sleep(0.1)
        GPIO.output(self.power_key_pin, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(self.power_key_pin, GPIO.LOW)
        time.sleep(18)
        self.ser.reset_input_buffer()

    def _send_at_command(self, command, expected_response) -> Union[str, None]:
        self.ser.write((command + '\r\n').encode())
        time.sleep(0.01)
        if self.ser.in_waiting:
            response = self.ser.read(self.ser.in_waiting).decode()
            if expected_response not in response or ',,,,,,,,' in response:
                return None
            else:
                return response.split(expected_response)[-1]
        else:
            return None

    @staticmethod
    def _convert_coordinates_to_gps(coordinates) -> tuple:
        parts = coordinates.split(',')

        latitude = float(parts[0])
        longitude = float(parts[2])
        lat_direction = parts[1]
        lon_direction = parts[3]
        latitude_degrees = latitude // 100 + (latitude % 100) / 60
        longitude_degrees = longitude // 100 + (longitude % 100) / 60

        if lat_direction == 'S':
            latitude_degrees *= -1
        if lon_direction == 'W':
            longitude_degrees *= -1

        return latitude_degrees, longitude_degrees


if __name__ == '__main__':
    gps = GPSModule()
    lat, lon = gps.get_gps_position()
    log.info(f'Latitude: {lat:.6f}, Longitude: {lon:.6f}')
    gps.power_down()
