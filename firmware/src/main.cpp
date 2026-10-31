#include <Arduino.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

namespace {
constexpr uint8_t kStepPin = 21;
constexpr uint8_t kDirectionPin = 22;
constexpr uint16_t kMinimumStepsPerSecond = 50;
constexpr uint16_t kMaximumStepsPerSecond = 1500;
constexpr uint32_t kCommandTimeoutMs = 400;
constexpr uint32_t kStepPulseHighUs = 3;
constexpr uint8_t kLeftDirectionLevel = LOW;
constexpr uint8_t kRightDirectionLevel = HIGH;

char commandBuffer[40] = {};
size_t commandLength = 0;
bool discardCommand = false;
bool moving = false;
bool movingLeft = false;
uint16_t stepsPerSecond = 100;
uint32_t stepIntervalUs = 1000000UL / stepsPerSecond;
uint32_t nextStepUs = 0;
uint32_t lastCommandMs = 0;

void stopMotor(bool report) {
  const bool wasMoving = moving;
  moving = false;
  digitalWrite(kStepPin, LOW);
  if (report && wasMoving) {
    Serial.println("OK STOP");
  }
}

void processCommand() {
  commandBuffer[commandLength] = '\0';
  for (size_t i = 0; i < commandLength; ++i) {
    commandBuffer[i] = static_cast<char>(toupper(static_cast<unsigned char>(commandBuffer[i])));
  }

  if (strcmp(commandBuffer, "STOP") == 0) {
    stopMotor(true);
    return;
  }

  char requestedDirection = 0;
  unsigned long requestedSpeed = 0;
  if (sscanf(commandBuffer, "JOG %c %lu", &requestedDirection, &requestedSpeed) != 2 ||
      (requestedDirection != 'L' && requestedDirection != 'R')) {
    if (commandLength > 0) {
      Serial.println("ERROR: USE JOG L|R <steps_per_second> OR STOP");
    }
    return;
  }

  requestedSpeed = constrain(requestedSpeed, kMinimumStepsPerSecond, kMaximumStepsPerSecond);
  const bool newMovingLeft = requestedDirection == 'L';
  const uint16_t newSpeed = static_cast<uint16_t>(requestedSpeed);
  const uint8_t directionLevel = newMovingLeft ? kLeftDirectionLevel : kRightDirectionLevel;
  const bool changed = !moving || movingLeft != newMovingLeft || stepsPerSecond != newSpeed;

  digitalWrite(kDirectionPin, directionLevel);
  if (changed) {
    movingLeft = newMovingLeft;
    stepsPerSecond = newSpeed;
    stepIntervalUs = 1000000UL / stepsPerSecond;
    nextStepUs = micros() + stepIntervalUs;
    Serial.print("OK JOG ");
    Serial.print(newMovingLeft ? "LEFT " : "RIGHT ");
    Serial.println(stepsPerSecond);
  }
  moving = true;
  lastCommandMs = millis();
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    const char incoming = static_cast<char>(Serial.read());
    if (incoming == '\r') {
      continue;
    }
    if (incoming == '\n') {
      if (discardCommand) {
        Serial.println("ERROR: COMMAND TOO LONG");
      } else {
        processCommand();
      }
      commandLength = 0;
      discardCommand = false;
      continue;
    }
    if (discardCommand) {
      continue;
    }
    if (commandLength + 1 < sizeof(commandBuffer)) {
      commandBuffer[commandLength++] = incoming;
    } else {
      discardCommand = true;
    }
  }
}

void generateStepPulse() {
  const uint32_t now = micros();
  if (static_cast<int32_t>(now - nextStepUs) < 0) {
    return;
  }

  digitalWrite(kStepPin, HIGH);
  delayMicroseconds(kStepPulseHighUs);
  digitalWrite(kStepPin, LOW);

  nextStepUs += stepIntervalUs;
  if (static_cast<int32_t>(now - nextStepUs) >= 0) {
    nextStepUs = now + stepIntervalUs;
  }
}
}  // namespace

void setup() {
  pinMode(kStepPin, OUTPUT);
  pinMode(kDirectionPin, OUTPUT);
  digitalWrite(kStepPin, LOW);
  digitalWrite(kDirectionPin, kLeftDirectionLevel);

  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("FIRE TOWER TMC2209 STEP/DIR READY");
  Serial.println("STEP=GPIO21 DIR=GPIO22; SPEED=50..1500 steps/s");
  Serial.println("Commands: JOG L|R <steps_per_second>, STOP");
  Serial.println("Motor remains stopped until a JOG command arrives.");
}

void loop() {
  readSerialCommands();

  if (moving && millis() - lastCommandMs > kCommandTimeoutMs) {
    stopMotor(true);
    Serial.println("WATCHDOG STOP");
  }

  if (moving) {
    generateStepPulse();
  }
}

