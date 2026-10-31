// ESP32-S3 capture -> desktop AquaSense server. No inference executes on the MCU.
// Hardware compilation, pin mapping and on-device tests remain pending.
#include <WiFi.h>
#include <HTTPClient.h>
#include "esp_camera.h"
#if __has_include("board_config.h")
#include "board_config.h"
#else
#include "board_config.example.h"
#endif
#if __has_include("secrets.h")
#include "secrets.h"
#else
#include "secrets.example.h"
#endif

bool ready = false;

bool connectWifi() {
  if (WiFi.status() == WL_CONNECTED) return true;
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) delay(250);
  return WiFi.status() == WL_CONNECTED;
}

void setup() {
  Serial.begin(115200);
  if (!CAMERA_PINS_CONFIGURED) {
    Serial.println("Configure board_config.h from your ESP32-S3 board schematic before camera initialization.");
    return;
  }
  const int requiredPins[] = {CAM_PIN_XCLK, CAM_PIN_SIOD, CAM_PIN_SIOC, CAM_PIN_D0,
    CAM_PIN_D1, CAM_PIN_D2, CAM_PIN_D3, CAM_PIN_D4, CAM_PIN_D5, CAM_PIN_D6,
    CAM_PIN_D7, CAM_PIN_VSYNC, CAM_PIN_HREF, CAM_PIN_PCLK};
  for (int pin : requiredPins) if (pin < 0) {Serial.println("Required camera pin is unset."); return;}
  if (!psramFound()) {Serial.println("This VGA capture configuration requires PSRAM."); return;}
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAM_PIN_D0; config.pin_d1 = CAM_PIN_D1;
  config.pin_d2 = CAM_PIN_D2; config.pin_d3 = CAM_PIN_D3;
  config.pin_d4 = CAM_PIN_D4; config.pin_d5 = CAM_PIN_D5;
  config.pin_d6 = CAM_PIN_D6; config.pin_d7 = CAM_PIN_D7;
  config.pin_xclk = CAM_PIN_XCLK; config.pin_pclk = CAM_PIN_PCLK;
  config.pin_vsync = CAM_PIN_VSYNC; config.pin_href = CAM_PIN_HREF;
  config.pin_sccb_sda = CAM_PIN_SIOD; config.pin_sccb_scl = CAM_PIN_SIOC;
  config.pin_pwdn = CAM_PIN_PWDN; config.pin_reset = CAM_PIN_RESET;
  config.xclk_freq_hz = CAMERA_XCLK_HZ;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 1;
  config.fb_location = CAMERA_FB_IN_PSRAM;
  config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t error = esp_camera_init(&config);
  if (error != ESP_OK) {Serial.printf("Camera initialization failed: 0x%x\n", error); return;}
  ready = true;
  Serial.println("Camera initialized. Place the full card in a fixed holder. Send c to capture.");
  Serial.println("Server estimates are synthetic-only and not validated for real water.");
}

void captureAndSend() {
  if (!connectWifi()) {Serial.println("Wi-Fi connection failed. Retry later."); return;}
  // Discard a few frames after scene changes; exposure stabilization must be validated on hardware.
  for (int i = 0; i < 3; ++i) {
    camera_fb_t* discarded = esp_camera_fb_get();
    if (discarded) esp_camera_fb_return(discarded);
    delay(150);
  }
  camera_fb_t* frame = esp_camera_fb_get();
  if (!frame) {Serial.println("Capture failed."); return;}
  if (frame->format != PIXFORMAT_JPEG || frame->width != 640 || frame->height != 480) {
    Serial.println("Server requires a 640x480 JPEG."); esp_camera_fb_return(frame); return;
  }
  WiFiClient client;
  HTTPClient http;
  http.setConnectTimeout(5000); http.setTimeout(20000);
  if (!http.begin(client, SERVER_URL)) {Serial.println("Invalid server URL."); esp_camera_fb_return(frame); return;}
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("X-API-Key", API_TOKEN);
  http.addHeader("X-Device", "esp32-s3");
  http.addHeader("X-Sample-Name", "ESP32-S3-capture");
  int status = http.POST(frame->buf, frame->len);
  esp_camera_fb_return(frame);
  if (status == 200) Serial.println("Image accepted; inspect per-metal status in response:");
  else if (status == 422) Serial.println("Image rejected; correct capture conditions:");
  else Serial.printf("HTTP/transport status: %d\n", status);
  Serial.println(http.getString());
  http.end();
}

void loop() {
  if (ready && Serial.available() && Serial.read() == 'c') captureAndSend();
  delay(20);
}
