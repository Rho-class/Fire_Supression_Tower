#include <Arduino.h>

// Fire Suppression Tower — ESP32 firmware entry point.
//
// Area owners: wire up your module here once it exists
// (ir-sensor / motor-control / pump-control). Keep this file thin —
// it should just call into each area's setup()/loop() hooks.

void setup() {
  Serial.begin(115200);
  Serial.println("Fire Suppression Tower firmware booting...");
}

void loop() {
  // TODO: IR sensor read, motor aim, pump control
}
