"""Autonomous firing: lock on, spiral the water, check, repeat until it is out.

    WATCH  -> the tracker centres the flame on the crosshair (CENTERED)
    LOCK   -> held CENTERED for LOCK_S (any drift restarts the wait)
    FIRE   -> pump ON, spray in rows (aim.RowSweep): the pan swings left and
              right across the flame, first at the flame's height, then the
              tilt creeps up very slowly to cover a farther flame; no
              self-correcting while spraying (open loop)
    CHECK  -> pump OFF, back on the flame line, CHECK_S for the view to clear,
              then WATCH again: a flame still there is re-centred and fired at.

Water: every sweep is one burst. Pump time is counted against a per-trial
budget (reservoir / trials, at PUMP_FLOW_ML_S); once a trial's share is used
it stops firing. A trial ends when the flame has been gone long enough for
the head to return home (call new_trial()).
"""

from __future__ import annotations

from aim import RowSweep
from config import (
    AUTOFIRE_CHECK_S, AUTOFIRE_LOCK_S, FIRE_FIRST_ROW_LEGS, FIRE_MAX_UP_DEG,
    FIRE_PAN_HALF_DEG, FIRE_PAN_SPEED_DEG_S, FIRE_RISE_DEG_S, PUMP_FLOW_ML_S,
    RESERVOIR_ML, TRIALS,
)


def make_row_sweep(left_deg: float) -> RowSweep:
    return RowSweep(left_deg, FIRE_PAN_HALF_DEG, FIRE_PAN_SPEED_DEG_S,
                    FIRE_FIRST_ROW_LEGS, FIRE_RISE_DEG_S, FIRE_MAX_UP_DEG)
from target_tracking import TrackingState


class AutoFire:
    def __init__(self, armed: bool = True, lock_s: float = AUTOFIRE_LOCK_S,
                 check_s: float = AUTOFIRE_CHECK_S, flow_ml_s: float = PUMP_FLOW_ML_S,
                 budget_ml: float = RESERVOIR_ML / TRIALS) -> None:
        self.armed = armed
        self.lock_s = lock_s
        self.check_s = check_s
        self.flow_ml_s = flow_ml_s
        self.budget_ml = budget_ml
        self.state = "WATCH"
        self.since = 0.0
        self.spiral: RowSweep | None = None  # the burst being fired
        self.trial_ml = 0.0  # water used this trial
        self.total_ml = 0.0  # water used this run
        self.bursts = 0  # bursts fired this trial
        self.last_now: float | None = None

    # -- water -------------------------------------------------------------
    @property
    def burst_ml(self) -> float:
        return make_row_sweep(0.0).duration * self.flow_ml_s

    @property
    def out_of_water(self) -> bool:
        """True once this trial's share is used (a burst already started finishes)."""
        return self.trial_ml >= self.budget_ml - 1e-6

    def new_trial(self) -> None:
        self.trial_ml, self.bursts = 0.0, 0

    # -- control -----------------------------------------------------------
    def disarm(self) -> None:
        self.armed = False
        self.spiral = None
        self.state = "WATCH"

    @property
    def firing(self) -> bool:
        return self.state == "FIRE"

    def update(self, now: float, tracking_state: TrackingState,
               aim_up_deg: float, aim_left_deg: float) -> tuple[tuple[float, float] | None, bool]:
        """Return ((pan, tilt) override or None for the tracker, pump wanted)."""
        dt = 0.0 if self.last_now is None else min(max(now - self.last_now, 0.0), 0.2)
        self.last_now = now
        if not self.armed:
            return None, False

        if self.state == "WATCH":
            if tracking_state == TrackingState.CENTERED and not self.out_of_water:
                self.state, self.since = "LOCK", now
            return None, False

        if self.state == "LOCK":
            if tracking_state != TrackingState.CENTERED:
                self.state = "WATCH"  # drifted off the crosshair: lock again
                return None, False
            if now - self.since < self.lock_s:
                return None, False
            self.state, self.since = "FIRE", now
            self.spiral = make_row_sweep(aim_left_deg)
            self.bursts += 1

        if self.state == "FIRE":
            assert self.spiral is not None
            command = self.spiral.update(now)
            pump = self.spiral.pump_wanted
            if pump:
                used = dt * self.flow_ml_s
                self.trial_ml += used
                self.total_ml += used
            if self.spiral.done:
                self.spiral = None
                self.state, self.since = "CHECK", now
                return (0.0, 0.0), False
            return command, pump

        if self.state == "CHECK":
            if now - self.since >= self.check_s:
                self.state = "WATCH"
            return (0.0, 0.0), False  # hold still while the view clears

        return None, False

    def status(self) -> str:
        water = f"water {self.trial_ml:.0f}/{self.budget_ml:.0f} mL"
        if not self.armed:
            return f"AUTO-FIRE OFF (SPACE = on)  {water}"
        if self.state == "FIRE" and self.spiral is not None:
            what = "aiming" if self.spiral.phase == "approach" else (
                "SPRAYING rows" if self.spiral.phase == "sweep" else "back to flame")
            return f"FIRE #{self.bursts}: {what}  {water}"
        if self.state == "LOCK":
            return f"LOCKED - firing in {max(self.lock_s - (self.last_now or 0) + self.since, 0):.1f} s  {water}"
        if self.state == "CHECK":
            return f"CHECKING if the flame is out  {water}"
        if self.out_of_water:
            return f"WATER SHARE FOR THIS TRIAL USED - not firing  {water}"
        return f"AUTO-FIRE ARMED  {water}"
