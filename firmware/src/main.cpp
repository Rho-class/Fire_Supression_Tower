#include <Arduino.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

namespace {
constexpr uint8_t kMotorCount = 2;
constexpr uint16_t kMinimumStepsPerSecond = 50;
constexpr uint16_t kMaximumStepsPerSecond = 1500;
constexpr uint32_t kCommandTimeoutMs = 400;
constexpr uint32_t kStepPulseHighUs = 3;
constexpr uint8_t kLeftDirectionLevel = LOW;
constexpr uint8_t kRightDirectionLevel = HIGH;

struct MotorState {
  uint8_t stepPin;
  uint8_t directionPin;
  bool moving;
  bool movingLeft;
  uint16_t stepsPerSecond;
  uint32_t stepIntervalUs;
  uint32_t nextStepUs;
  uint32_t lastCommandMs;
};

MotorState motors[kMotorCount] = {
    {21, 22, false, false, 100, 10000, 0, 0},
    {18, 19, false, false, 100, 10000, 0, 0},
};
uint8_t legacyMotorIndex = 0;
char commandBuffer[40] = {};
size_t commandLength = 0;
bool discardCommand = false;

void stopMotor(MotorState& motor, bool report) {
  const bool wasMoving = motor.moving;
  motor.moving = false;
  digitalWrite(motor.stepPin, LOW);
  if (report && wasMoving) {
    Serial.println("OK STOP");
  }
}

void stopAllMotors() {
  for (MotorState& motor : motors) {
    stopMotor(motor, false);
  }
  Serial.println("OK STOP");
}

void jogMotor(MotorState& motor, char requestedDirection, unsigned long requestedSpeed) {
  requestedSpeed = constrain(requestedSpeed, kMinimumStepsPerSecond, kMaximumStepsPerSecond);
  const bool newMovingLeft = requestedDirection == 'L';
  const uint16_t newSpeed = static_cast<uint16_t>(requestedSpeed);
  const uint8_t directionLevel = newMovingLeft ? kLeftDirectionLevel : kRightDirectionLevel;
  const bool changed = !motor.moving || motor.movingLeft != newMovingLeft || motor.stepsPerSecond != newSpeed;

  digitalWrite(motor.directionPin, directionLevel);
  if (changed) {
    motor.movingLeft = newMovingLeft;
    motor.stepsPerSecond = newSpeed;
    motor.stepIntervalUs = 1000000UL / motor.stepsPerSecond;
    motor.nextStepUs = micros() + motor.stepIntervalUs;
    Serial.print("OK JOG ");
    Serial.print(newMovingLeft ? "LEFT " : "RIGHT ");
    Serial.println(motor.stepsPerSecond);
  }
  motor.moving = true;
  motor.lastCommandMs = millis();
}

void processCommand() {
  commandBuffer[commandLength] = '\0';
  for (size_t i = 0; i < commandLength; ++i) {
    commandBuffer[i] = static_cast<char>(toupper(static_cast<unsigned char>(commandBuffer[i])));
  }

  if (strcmp(commandBuffer, "STOP") == 0) {
    stopAllMotors();
    return;
  }

  unsigned int motorNumber = 0;
  if (sscanf(commandBuffer, "STOP %u", &motorNumber) == 1) {
    if (motorNumber < 1 || motorNumber > kMotorCount) {
      Serial.println("ERROR: MOTOR MUST BE 1 OR 2");
      return;
    }
    stopMotor(motors[motorNumber - 1], true);
    return;
  }

  unsigned int requestedStepPin = 0;
  unsigned int requestedDirectionPin = 0;
  if (sscanf(commandBuffer, "PINS %u %u", &requestedStepPin, &requestedDirectionPin) == 2) {
    if (requestedStepPin == 21 && requestedDirectionPin == 22) {
      legacyMotorIndex = 0;
    } else if (requestedStepPin == 18 && requestedDirectionPin == 19) {
      legacyMotorIndex = 1;
    } else {
      Serial.println("ERROR: PINS MUST BE 21 22 OR 18 19");
      return;
    }
    stopMotor(motors[legacyMotorIndex], false);
    Serial.print("OK PINS STEP=GPIO");
    Serial.print(motors[legacyMotorIndex].stepPin);
    Serial.print(" DIR=GPIO");
    Serial.println(motors[legacyMotorIndex].directionPin);
    return;
  }

  char requestedDirection = 0;
  unsigned long requestedSpeed = 0;
  if (sscanf(commandBuffer, "JOG %u %c %lu", &motorNumber, &requestedDirection, &requestedSpeed) == 3) {
    if (motorNumber < 1 || motorNumber > kMotorCount ||
        (requestedDirection != 'L' && requestedDirection != 'R')) {
      Serial.println("ERROR: USE JOG 1|2 L|R <steps_per_second>");
      return;
    }
    jogMotor(motors[motorNumber - 1], requestedDirection, requestedSpeed);
    return;
  }

  if (sscanf(commandBuffer, "JOG %c %lu", &requestedDirection, &requestedSpeed) == 2 &&
      (requestedDirection == 'L' || requestedDirection == 'R')) {
    jogMotor(motors[legacyMotorIndex], requestedDirection, requestedSpeed);
    return;
  }

  if (commandLength > 0) {
    Serial.println("ERROR: USE JOG 1|2 L|R <speed>, JOG L|R <speed>, OR STOP");
  }
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

void generateStepPulse(MotorState& motor) {
  const uint32_t now = micros();
  if (static_cast<int32_t>(now - motor.nextStepUs) < 0) {
    return;
  }

  digitalWrite(motor.stepPin, HIGH);
  delayMicroseconds(kStepPulseHighUs);
  digitalWrite(motor.stepPin, LOW);

  motor.nextStepUs += motor.stepIntervalUs;
  if (static_cast<int32_t>(now - motor.nextStepUs) >= 0) {
    motor.nextStepUs = now + motor.stepIntervalUs;
  }
}
}  // namespace

void setup() {
  for (MotorState& motor : motors) {
    pinMode(motor.stepPin, OUTPUT);
    pinMode(motor.directionPin, OUTPUT);
    digitalWrite(motor.stepPin, LOW);
    digitalWrite(motor.directionPin, kLeftDirectionLevel);
  }

  Serial.begin(115200);
  delay(300);
  Serial.println();
  Serial.println("FIRE TOWER DUAL TMC2209 STEP/DIR READY");
  Serial.println("MOTOR1 STEP=GPIO21 DIR=GPIO22; MOTOR2 STEP=GPIO18 DIR=GPIO19");
  Serial.println("Commands: JOG 1|2 L|R <steps_per_second>, STOP 1|2, STOP");
  Serial.println("Legacy PINS and JOG commands are supported; motors start stopped.");
}

void loop() {
  readSerialCommands();

  const uint32_t nowMs = millis();
  for (MotorState& motor : motors) {
    if (motor.moving && nowMs - motor.lastCommandMs > kCommandTimeoutMs) {
      stopMotor(motor, true);
      Serial.println("WATCHDOG STOP");
    }
    if (motor.moving) {
      generateStepPulse(motor);
    }
  }
}
