# ESP32-S3 capture client — hardware verification pending

The sketch sends a VGA JPEG to the desktop server and prints its JSON response. It does not run Python models on the ESP32-S3. No exact camera board or sensor module has been identified, so the camera pin map is intentionally unset. The sketch refuses to initialize the camera until you configure it.

1. Identify the exact board model, camera sensor, PSRAM configuration and vendor pin map.
2. Open `aquasense_s3/aquasense_s3.ino` using the Arduino ESP32 board package matching your hardware.
3. Copy `board_config.example.h` to `board_config.h`, enter vendor pins/settings, and set `CAMERA_PINS_CONFIGURED` to true. Do not infer pins from another ESP32-S3 board.
4. Copy `secrets.example.h` to `secrets.h`; fill Wi-Fi credentials, the PC's LAN server address, and the server token.
5. On a trusted local network, set `AQUASENSE_API_TOKEN` in the PC environment and run `python app.py --host 0.0.0.0`. Permit the specific port through the firewall only if required by your network setup.
6. Compile and upload for the exact board. Monitor serial at 115200 baud; send `c` to capture. HTTP 422 means the capture was rejected and includes a reason.

The HTTP transport is for an isolated lab LAN, not an internet deployment. The current sketch requires PSRAM and a sensor supporting JPEG at VGA. Capture, illumination, exposure, white-balance settings, memory, latency, power and reconnection behavior must be measured on the real board. No Arduino toolchain or device is available in this workspace, so the sketch has not been compiled or flashed here.

The interface and configuration names were checked against Espressif's official [camera driver](https://github.com/espressif/esp32-camera/blob/master/driver/include/esp_camera.h) and [board-specific camera pin definitions](https://github.com/espressif/arduino-esp32/blob/master/libraries/ESP32/examples/Camera/CameraWebServer/camera_pins.h). These references show why an ESP32-S3 chip name alone does not identify the pin map.
