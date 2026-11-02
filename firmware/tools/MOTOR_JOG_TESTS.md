# TMC2209 keyboard motor tests

All test programs use the ESP32 serial connection (default `COM3`) and the same TMC2209 STEP/DIR protocol. The firmware defaults to motor 1 and accepts only these pin pairs:

| Motor | STEP | DIR |
| --- | --- | --- |
| 1 | GPIO 21 | GPIO 22 |
| 2 | GPIO 18 | GPIO 19 |

Flash the updated firmware from the `firmware/` directory before running a test. The combined program controls both motors independently; the single-motor programs remain available for separate testing.

Run both motors from one window:

```powershell
python firmware/tools/keyboard_jog_both_motors.py
```

Motor 1 uses left/right arrows to jog and up/down arrows to change its speed. Motor 2 uses A/D to jog and W/S to change its speed. Both motors can move at once.

Run only motor 1:

```powershell
python firmware/tools/keyboard_jog.py
```

Run only motor 2:

```powershell
python firmware/tools/keyboard_jog_motor2.py
```

Pass `--port COM4` (or the detected port) if the ESP32 is not on `COM3`. Hold left or right to jog, release to stop, use up/down to adjust speed, and press `Q` or Escape to quit. Speed starts at 100 steps/s and is limited to 50–500 steps/s.
