#pragma once
// Copy to board_config.h, fill from YOUR board vendor's schematic, then set true.
// ESP32-S3 is a chip family, not a camera-pin mapping. No board pins are assumed.
constexpr bool CAMERA_PINS_CONFIGURED = false;
constexpr int CAM_PIN_PWDN = -1;
constexpr int CAM_PIN_RESET = -1;
constexpr int CAM_PIN_XCLK = -1;
constexpr int CAM_PIN_SIOD = -1;
constexpr int CAM_PIN_SIOC = -1;
constexpr int CAM_PIN_D0 = -1;
constexpr int CAM_PIN_D1 = -1;
constexpr int CAM_PIN_D2 = -1;
constexpr int CAM_PIN_D3 = -1;
constexpr int CAM_PIN_D4 = -1;
constexpr int CAM_PIN_D5 = -1;
constexpr int CAM_PIN_D6 = -1;
constexpr int CAM_PIN_D7 = -1;
constexpr int CAM_PIN_VSYNC = -1;
constexpr int CAM_PIN_HREF = -1;
constexpr int CAM_PIN_PCLK = -1;
constexpr unsigned long CAMERA_XCLK_HZ = 20000000;
// Confirm XCLK and sensor settings against the actual camera module datasheet.
