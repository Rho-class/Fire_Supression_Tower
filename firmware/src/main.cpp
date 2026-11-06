// ESP32 + 2x TMC2208 stepper jog controller (two independent motors) + pump,
// controlled over Wi-Fi from a web page (and still over USB serial).
//
// Wiring (see the diagrams from our chat):
//   Motor 1:  ESP32 D18 -> driver 1 STEP,   ESP32 D19 -> driver 1 DIR
//   Motor 2:  ESP32 D21 -> driver 2 STEP,   ESP32 D22 -> driver 2 DIR
//   Both drivers: VDD -> ESP32 3V3,  GND / EN / PDN -> ESP32 GND
//   MS1 / MS2 left open on both -> 1/8 microstepping
//   Pump:     ESP32 D23 -> MOSFET trigger module TRIG (module GND -> ESP32 GND)
//
// How to connect (with the default settings below):
//   1. Join the Wi-Fi network "StepperControl" (password below) on your laptop or phone.
//   2. Open http://192.168.4.1 in a browser.
//   Keys: A/D turn motor 1, W/S its speed; Left/Right turn motor 2, Up/Down its speed;
//   E toggles the pump; Space stops everything. The page also has touch buttons.
//
// Commands (same one-letter commands over Wi-Fi and USB serial at 115200 baud):
//   L / R  motor 1 left / right        B / F  motor 2 back / forward
//   X / Y  stop motor 1 / motor 2      S  stop everything (motors + pump)
//   U / D  motor 1 speed up / down     P / M  motor 2 speed up / down
//   E      toggle pump on / off        K  pump off
//   .      heartbeat (no action, just "the controller is still here")
//   Z      make the current position the centre (both motors must be stopped)
//   ?      print status and pan/tilt position
//
// Travel limits: every jog stops at +/- PAN_LIMIT_DEG / TILT_LIMIT_DEG from
// the power-on (or 'Z') position, so the head cannot be driven into its stops.
// Power on with the head pointing straight ahead and level. "LIMIT pan/tilt
// reached" is printed when a jog stops at a limit.
//
// Laptop flame tracker (newline-terminated USB / WebSocket text):
//   PAN:<float>,TILT:<float>
//   pan < 0 -> motor 1 left;  pan > 0 -> motor 1 right;  pan == 0 -> stop M1
//   tilt < 0 -> motor 2 back (up); tilt > 0 -> motor 2 forward (down); 0 -> stop M2
//   Speed scales with |pan| / |tilt| from TRACK_MIN_RPM to TRACK_MAX_RPM (over USB serial).
//   The laptop must keep sending while tracking so JOG_TIMEOUT_MS does not stop the motors.
//
// Safety:
//   - Each jog only lasts JOG_TIMEOUT_MS after its last command; the page resends
//     every 100 ms while a button or key is held, so motors stop if the connection drops.
//   - The pump turns off if no command or heartbeat arrives for PUMP_TIMEOUT_MS.
//   - Closing the page (or the Wi-Fi dropping) stops everything.
//
// The motors are stepped on CPU core 1 (Arduino loop) while Wi-Fi runs on core 0,
// so network traffic doesn't make the motors stutter.

#include <Arduino.h>
#include <AccelStepper.h>
#include <WiFi.h>
#include <WebServer.h>
#include <WebSocketsServer.h>
#include <ESPmDNS.h>
#include "web_page.h"

// ---------- Wi-Fi ----------
// true:  the ESP32 makes its own network (works anywhere, no router needed).
// false: the ESP32 joins your home Wi-Fi; open http://stepper.local (or the IP
//        printed in the Serial Monitor). Falls back to its own network if it can't join.
constexpr bool OWN_NETWORK = true;
const char *AP_SSID   = "StepperControl";
const char *AP_PASS   = "change-me-123";   // at least 8 characters; change it
const char *HOME_SSID = "your-wifi-name";
const char *HOME_PASS = "your-wifi-password";
const char *HOSTNAME  = "stepper";         // -> http://stepper.local on home Wi-Fi

// ---------- Pins ----------
constexpr uint8_t M1_STEP_PIN = 18;
constexpr uint8_t M1_DIR_PIN  = 19;
constexpr uint8_t M2_STEP_PIN = 21;
constexpr uint8_t M2_DIR_PIN  = 22;
constexpr uint8_t PUMP_PIN    = 23;         // HIGH = pump on

// ---------- Motor setup ----------
constexpr int   FULL_STEPS_PER_REV = 200;   // 1.8 degree motor
constexpr int   MICROSTEPS         = 8;     // MS1/MS2 open on TMC2208
constexpr float STEPS_PER_REV      = FULL_STEPS_PER_REV * MICROSTEPS;  // 1600

// Set true for a motor that turns the wrong way.
constexpr bool M1_INVERT_DIRECTION = false;
constexpr bool M2_INVERT_DIRECTION = false;

// Flip the vision PAN:/TILT: axis sense without rewiring (after laptop invert flags).
constexpr bool PAN_AXIS_INVERT  = false;
constexpr bool TILT_AXIS_INVERT = false;

constexpr float PAN_TILT_EPS = 0.001f;

// ---------- Travel limits and tracking speed ----------
// Both axes have physical limits (frame / cables), so every jog (web, serial
// letters, or laptop PAN/TILT tracking) stops at +/- the limit below, measured
// from the position at power-on or after the 'Z' command. Power on, or press
// 'Z', with the head pointing straight ahead and level.
//
// *_GEAR_RATIO = motor turns per one turn of the head. Left at 1.0, limits and
// speeds are motor-shaft values; with a reduction gear or belt the head moves
// LESS far and slower than these numbers (the safe direction). Set the real
// ratio once it is known so the limits become head degrees.
constexpr float PAN_GEAR_RATIO  = 1.0f;
constexpr float TILT_GEAR_RATIO = 1.0f;
constexpr float PAN_LIMIT_DEG   = 45.0f;    // each side of centre
constexpr float TILT_LIMIT_DEG  = 20.0f;    // up and down from level

// Laptop tracking speed scales with the size of the PAN/TILT correction
// (0..TRACK_FULL_COMMAND, the laptop's MAX_PAN/TILT_COMMAND): slow near the
// target so the head does not overshoot, faster when the flame is far off.
constexpr float TRACK_MIN_RPM      = 2.0f;
constexpr float TRACK_MAX_RPM      = 10.0f;
constexpr float TRACK_FULL_COMMAND = 25.0f;

// Speed presets in revolutions per minute, selected with speed up / down.
const float SPEED_RPM[] = {15, 30, 60, 90, 120, 180, 360, 720};
constexpr int SPEED_COUNT   = sizeof(SPEED_RPM) / sizeof(SPEED_RPM[0]);
constexpr int DEFAULT_SPEED = 2;            // 60 RPM

constexpr float ACCEL_STEPS_PER_S2 = 30000;  // smooth start / stop
constexpr uint32_t JOG_TIMEOUT_MS  = 500;   // stop a motor if no jog command for this long
constexpr uint32_t PUMP_TIMEOUT_MS = 1500;  // pump off if the controller goes quiet

// ---------- State ----------
struct Motor {
  AccelStepper stepper;
  int      direction  = 0;                  // -1, 0 stopped, +1
  uint32_t lastJogMs  = 0;
  int      speedIndex = DEFAULT_SPEED;      // each motor has its own speed
  long     limitSteps = 0;                  // soft travel limit, +/- from zero
  bool     limitReported = false;
  Motor(uint8_t stepPin, uint8_t dirPin) : stepper(AccelStepper::DRIVER, stepPin, dirPin) {}
};

Motor motors[2] = {Motor(M1_STEP_PIN, M1_DIR_PIN), Motor(M2_STEP_PIN, M2_DIR_PIN)};
bool     pumpOn   = false;
uint32_t lastRxMs = 0;                      // time of the last command from anywhere

// Commands from Wi-Fi (core 0) are handed to the motor loop (core 1) through this queue.
QueueHandle_t commandQueue;

// Latest status line, written by the motor loop and broadcast by the Wi-Fi task.
portMUX_TYPE statusLock = portMUX_INITIALIZER_UNLOCKED;
char statusLine[64] = "";
bool statusChanged  = false;

WebServer        http(80);
WebSocketsServer ws(81);

// USB serial line assembly for PAN:/TILT: (and newline-terminated letter batches).
char     serialLine[96];
size_t   serialLineLen = 0;
// When 'P'/'p' arrives, peek whether this is "PAN:..." vs motor-2 speed-up.
char     pendingChars[96];
size_t   pendingLen = 0;
uint32_t pendingStartedMs = 0;
constexpr uint32_t PENDING_P_TIMEOUT_MS = 30;

// ---------- Motor / pump logic (runs on core 1) ----------
float rpmToStepsPerSec(float rpm) { return rpm * STEPS_PER_REV / 60.0f; }

void applySpeed(Motor &m) {
  m.stepper.setMaxSpeed(rpmToStepsPerSec(SPEED_RPM[m.speedIndex]));
}

void changeSpeed(Motor &m, int delta) {
  m.speedIndex = constrain(m.speedIndex + delta, 0, SPEED_COUNT - 1);
  applySpeed(m);
}

const char *stateName(int direction) {
  return direction < 0 ? "left" : direction > 0 ? "right" : "stopped";
}

// Format: STATUS <m1 state> <m1 rpm> <m2 state> <m2 rpm> <pump on/off>
void printStatus() {
  char line[64];
  snprintf(line, sizeof(line), "STATUS %s %.0f %s %.0f %s",
           stateName(motors[0].direction), SPEED_RPM[motors[0].speedIndex],
           stateName(motors[1].direction), SPEED_RPM[motors[1].speedIndex],
           pumpOn ? "on" : "off");
  Serial.println(line);
  portENTER_CRITICAL(&statusLock);
  strcpy(statusLine, line);
  statusChanged = true;
  portEXIT_CRITICAL(&statusLock);
}

void setPump(bool on) {
  if (on == pumpOn) return;
  pumpOn = on;
  digitalWrite(PUMP_PIN, on ? HIGH : LOW);
  printStatus();
}

void startJog(Motor &m, int direction) {
  m.lastJogMs = millis();
  if (direction == m.direction) return;     // already moving this way
  m.direction = direction;
  // Aim at the soft limit in that direction; stop() decelerates sooner if the
  // jog ends first. The motor can never be driven past +/- limitSteps.
  m.stepper.moveTo((long)direction * m.limitSteps);
  printStatus();
}

void stopJog(Motor &m);  // defined below

// Manual jog (web page, keyboard tools, serial letters): preset speed.
void manualJog(Motor &m, int direction) {
  applySpeed(m);
  startJog(m, direction);
}

// Laptop tracking jog: speed proportional to the correction's size.
void trackAxis(Motor &m, float command) {
  float magnitude = fabsf(command);
  if (magnitude <= PAN_TILT_EPS) {
    stopJog(m);
    return;
  }
  float fraction = magnitude >= TRACK_FULL_COMMAND ? 1.0f : magnitude / TRACK_FULL_COMMAND;
  float rpm = TRACK_MIN_RPM + (TRACK_MAX_RPM - TRACK_MIN_RPM) * fraction;
  m.stepper.setMaxSpeed(rpmToStepsPerSec(rpm));
  startJog(m, command < 0 ? -1 : +1);
}

long limitStepsFor(float limitDeg, float gearRatio) {
  return (long)(limitDeg / 360.0f * STEPS_PER_REV * gearRatio);
}

// 'Z': make the current position the centre. Only while both motors are idle.
void zeroPositions() {
  for (Motor &m : motors) {
    if (m.direction != 0 || m.stepper.distanceToGo() != 0) {
      Serial.println("ZERO refused: stop both motors first");
      return;
    }
  }
  for (Motor &m : motors) m.stepper.setCurrentPosition(0);
  Serial.println("ZERO current position is now centre");
}

void printPositions() {
  Serial.printf("POS pan %.1f deg (limit %.0f), tilt %.1f deg (limit %.0f)\n",
                motors[0].stepper.currentPosition() * 360.0f / (STEPS_PER_REV * PAN_GEAR_RATIO),
                PAN_LIMIT_DEG,
                motors[1].stepper.currentPosition() * 360.0f / (STEPS_PER_REV * TILT_GEAR_RATIO),
                TILT_LIMIT_DEG);
}

void stopJog(Motor &m) {
  if (m.direction == 0) return;
  m.direction = 0;
  m.stepper.stop();                         // decelerate, don't slam
  printStatus();
}

void handleCommand(char c) {
  lastRxMs = millis();
  switch (c) {
    case 'L': case 'l': manualJog(motors[0], -1); break;
    case 'R': case 'r': manualJog(motors[0], +1); break;
    case 'B': case 'b': manualJog(motors[1], -1); break;
    case 'F': case 'f': manualJog(motors[1], +1); break;
    case 'Z': case 'z': zeroPositions(); break;
    case 'X': case 'x': stopJog(motors[0]); break;
    case 'Y': case 'y': stopJog(motors[1]); break;
    case 'S': case 's': stopJog(motors[0]); stopJog(motors[1]); setPump(false); break;
    case 'E': case 'e': setPump(!pumpOn); break;
    case 'K': case 'k': setPump(false); break;
    case 'U': case 'u': changeSpeed(motors[0], +1); printStatus(); break;
    case 'D': case 'd': changeSpeed(motors[0], -1); printStatus(); break;
    case 'P': case 'p': changeSpeed(motors[1], +1); printStatus(); break;
    case 'M': case 'm': changeSpeed(motors[1], -1); printStatus(); break;
    case '?': printStatus(); printPositions(); break;
    default: break;                         // '.' heartbeat, newlines etc.
  }
}

// Map laptop PAN:/TILT: corrections to jog letters. Does not touch the pump.
void handlePanTiltLine(const char *line) {
  float pan = 0.0f;
  float tilt = 0.0f;
  if (sscanf(line, "PAN:%f,TILT:%f", &pan, &tilt) != 2) {
    return;
  }
  if (PAN_AXIS_INVERT) {
    pan = -pan;
  }
  if (TILT_AXIS_INVERT) {
    tilt = -tilt;
  }

  lastRxMs = millis();
  trackAxis(motors[0], pan);               // < 0 left, > 0 right, 0 stop
  trackAxis(motors[1], tilt);              // < 0 up (back), > 0 down (forward)
}

bool looksLikePanTilt(const char *text, size_t len) {
  static const char prefix[] = "PAN:";
  if (len < 4) {
    return false;
  }
  for (size_t i = 0; i < 4; i++) {
    if (text[i] != prefix[i]) {
      return false;
    }
  }
  return true;
}

void handleTextPayload(const char *text, size_t len) {
  if (looksLikePanTilt(text, len)) {
    char line[96];
    size_t copyLen = len < sizeof(line) - 1 ? len : sizeof(line) - 1;
    memcpy(line, text, copyLen);
    line[copyLen] = '\0';
    // Trim trailing CR/LF for sscanf.
    while (copyLen > 0 && (line[copyLen - 1] == '\n' || line[copyLen - 1] == '\r')) {
      line[--copyLen] = '\0';
    }
    // Queue a marker + line via one-char queue: process on motor core by
    // enqueueing synthetic letters after parse on this core is unsafe for
    // AccelStepper. Instead enqueue each resulting command letter.
    float pan = 0.0f;
    float tilt = 0.0f;
    if (sscanf(line, "PAN:%f,TILT:%f", &pan, &tilt) != 2) {
      return;
    }
    if (PAN_AXIS_INVERT) {
      pan = -pan;
    }
    if (TILT_AXIS_INVERT) {
      tilt = -tilt;
    }
    char cmds[4];
    size_t n = 0;
    cmds[n++] = pan < -PAN_TILT_EPS ? 'L' : pan > PAN_TILT_EPS ? 'R' : 'X';
    cmds[n++] = tilt < -PAN_TILT_EPS ? 'B' : tilt > PAN_TILT_EPS ? 'F' : 'Y';
    for (size_t i = 0; i < n; i++) {
      xQueueSend(commandQueue, &cmds[i], 0);
    }
    return;
  }
  for (size_t i = 0; i < len; i++) {
    xQueueSend(commandQueue, &text[i], 0);
  }
}

void flushPendingAsCommands() {
  for (size_t i = 0; i < pendingLen; i++) {
    handleCommand(pendingChars[i]);
  }
  pendingLen = 0;
}

void feedSerialByte(char c) {
  if (pendingLen > 0) {
    if (pendingLen < sizeof(pendingChars)) {
      pendingChars[pendingLen++] = c;
    } else {
      flushPendingAsCommands();
      return;
    }
    if (c == '\n' || c == '\r') {
      pendingChars[pendingLen] = '\0';
      if (looksLikePanTilt(pendingChars, pendingLen)) {
        while (pendingLen > 0 &&
               (pendingChars[pendingLen - 1] == '\n' ||
                pendingChars[pendingLen - 1] == '\r')) {
          pendingChars[--pendingLen] = '\0';
        }
        handlePanTiltLine(pendingChars);
        pendingLen = 0;
      } else {
        flushPendingAsCommands();
      }
      return;
    }
    // Still matching "PAN:" prefix?
    static const char prefix[] = "PAN:";
    if (pendingLen <= 4) {
      for (size_t i = 0; i < pendingLen; i++) {
        if (pendingChars[i] != prefix[i]) {
          flushPendingAsCommands();
          return;
        }
      }
    }
    return;
  }

  if (c == 'P' || c == 'p') {
    pendingChars[0] = c;
    pendingLen = 1;
    pendingStartedMs = millis();
    return;
  }

  if (c == '\n' || c == '\r') {
    if (serialLineLen > 0) {
      serialLine[serialLineLen] = '\0';
      if (looksLikePanTilt(serialLine, serialLineLen)) {
        handlePanTiltLine(serialLine);
      } else {
        for (size_t i = 0; i < serialLineLen; i++) {
          handleCommand(serialLine[i]);
        }
      }
      serialLineLen = 0;
    }
    return;
  }

  handleCommand(c);
}

// ---------- Wi-Fi (runs on core 0) ----------
void queueCommands(const char *text, size_t len) {
  handleTextPayload(text, len);
}

void onWebSocket(uint8_t client, WStype_t type, uint8_t *payload, size_t length) {
  switch (type) {
    case WStype_CONNECTED: {
      char q = '?';
      xQueueSend(commandQueue, &q, 0);
      break;
    }
    case WStype_TEXT:
      handleTextPayload((const char *)payload, length);
      break;
    case WStype_DISCONNECTED: {
      char s = 'S';
      xQueueSend(commandQueue, &s, 0);
      break;
    }
    default:
      break;
  }
}

void startWiFi() {
  WiFi.setSleep(false);                     // lower, steadier latency for the controls
  bool joined = false;
  if (!OWN_NETWORK) {
    WiFi.mode(WIFI_STA);
    WiFi.setHostname(HOSTNAME);
    WiFi.begin(HOME_SSID, HOME_PASS);
    Serial.printf("Joining Wi-Fi \"%s\"", HOME_SSID);
    for (int i = 0; i < 30 && WiFi.status() != WL_CONNECTED; i++) {
      delay(500);
      Serial.print('.');
    }
    Serial.println();
    joined = WiFi.status() == WL_CONNECTED;
    if (joined) {
      Serial.printf("Connected. Open http://%s.local or http://%s\n",
                    HOSTNAME, WiFi.localIP().toString().c_str());
      if (MDNS.begin(HOSTNAME)) MDNS.addService("http", "tcp", 80);
    } else {
      Serial.println("Couldn't join; starting own network instead.");
    }
  }
  if (!joined) {
    WiFi.mode(WIFI_AP);
    WiFi.softAP(AP_SSID, AP_PASS);
    Serial.printf("Wi-Fi network \"%s\" ready. Join it, then open http://%s\n",
                  AP_SSID, WiFi.softAPIP().toString().c_str());
  }

  http.on("/", []() { http.send_P(200, "text/html", WEB_PAGE); });
  http.onNotFound([]() { http.send(404, "text/plain", "Not found"); });
  http.begin();
  ws.begin();
  ws.onEvent(onWebSocket);
}

void networkTask(void *) {
  char line[64];
  for (;;) {
    http.handleClient();
    ws.loop();

    bool send = false;
    portENTER_CRITICAL(&statusLock);
    if (statusChanged) {
      strcpy(line, statusLine);
      statusChanged = false;
      send = true;
    }
    portEXIT_CRITICAL(&statusLock);
    if (send) ws.broadcastTXT(line);

    vTaskDelay(1);                          // let the Wi-Fi stack run
  }
}

// ---------- Setup / loop ----------
void setup() {
  digitalWrite(PUMP_PIN, LOW);              // pump off before anything else
  pinMode(PUMP_PIN, OUTPUT);
  Serial.begin(115200);
  motors[0].stepper.setPinsInverted(M1_INVERT_DIRECTION, false, false);
  motors[1].stepper.setPinsInverted(M2_INVERT_DIRECTION, false, false);
  motors[0].limitSteps = limitStepsFor(PAN_LIMIT_DEG, PAN_GEAR_RATIO);
  motors[1].limitSteps = limitStepsFor(TILT_LIMIT_DEG, TILT_GEAR_RATIO);
  for (Motor &m : motors) {
    m.stepper.setMinPulseWidth(2);          // microseconds; TMC2208 needs ~0.1
    m.stepper.setAcceleration(ACCEL_STEPS_PER_S2);
    applySpeed(m);
  }
  delay(200);
  Serial.println("READY stepper jog controller (2 motors + pump, Wi-Fi, PAN/TILT)");

  commandQueue = xQueueCreate(128, sizeof(char));
  startWiFi();
  xTaskCreatePinnedToCore(networkTask, "network", 8192, nullptr, 1, nullptr, 0);
  printStatus();
}

void loop() {
  while (Serial.available()) {
    feedSerialByte((char)Serial.read());
  }
  // Lone serial 'P'/'p' (speed up) without following "AN:" — flush after a short wait.
  if (pendingLen == 1 &&
      (pendingChars[0] == 'P' || pendingChars[0] == 'p') &&
      (millis() - pendingStartedMs) >= PENDING_P_TIMEOUT_MS) {
    flushPendingAsCommands();
  }
  char c;
  while (xQueueReceive(commandQueue, &c, 0) == pdTRUE) {
    handleCommand(c);
  }

  uint32_t now = millis();
  if (pumpOn && now - lastRxMs > PUMP_TIMEOUT_MS) setPump(false);

  for (int i = 0; i < 2; i++) {
    Motor &m = motors[i];
    if (m.direction != 0 && now - m.lastJogMs > JOG_TIMEOUT_MS) stopJog(m);
    m.stepper.run();                        // must run as often as possible
    bool atLimit = m.direction != 0 && m.stepper.distanceToGo() == 0;
    if (atLimit && !m.limitReported) {
      Serial.printf("LIMIT %s reached\n", i == 0 ? "pan" : "tilt");
    }
    m.limitReported = atLimit;
  }
}
