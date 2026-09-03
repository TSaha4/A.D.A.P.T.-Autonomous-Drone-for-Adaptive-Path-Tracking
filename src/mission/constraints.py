"""
Battery / payload / energy model for the mission planner (Task 3).

The model is *demo-grade*: it is a transparent linear model anchored on a real
consumer drone spec (DJI Mavic-class) so the constants are documented and
sourced, not arbitrary numbers that pretend to be certified telemetry.

Sourced anchors (DJI Mavic 3)
-----------------------------
* Battery: 5000 mAh / 77 Wh Intelligent Flight Battery.
* Max flight time: ~46 min (hover, no wind).
* Max horizontal speed: 21 m/s (Sport mode).

  Sources:
    - DJI Mavic 3 specs  https://www.dji.com/mavic-3/specs
    - DJI Mavic (overview)  https://en.wikipedia.org/wiki/DJI_Mavic

Everything else (cruise speed, per-kg payload penalty, wind-power coefficient,
capacity in kg, reserve fraction) is an openly-labelled configuration for this
demo -- a Mavic-class *consumer* quadcopter can in reality carry almost no
relief payload, so operators should retune :class:`DroneSpec` for the actual
airframe.  Change the values, not the physics structure.
"""
import math
from dataclasses import dataclass, field
from typing import Dict, Optional

# --------------------------------------------------------------------------- #
# Specs (with sourced values as defaults)
# --------------------------------------------------------------------------- #
@dataclass
class DroneSpec:
    # ---- sourced from DJI Mavic 3 -------------------------------- #
    battery_wh: float = 77.0          # 5000 mAh x 15.4 V (DJI spec)
    hover_endurance_min: float = 46.0  # DJI spec (hover, no wind)
    max_speed_mps: float = 21.0        # DJI spec (Sport mode)
    # ---- demo configuration (clearly labelled) -------------------- #
    cruise_airspeed_mps: float = 12.0  # economical cruise
    min_speed_mps: float = 4.0         # cannot hold station below this
    payload_capacity_kg: float = 2.0   # demo; retune for real airframe
    payload_energy_per_kg: float = 0.15   # +15% power per kg carried (demo)
    wind_energy_coef: float = 0.8         # extra energy into a headwind (demo)
    cruise_power_factor: float = 1.2      # cruise draws more than hover (demo)
    usable_battery_fraction: float = 0.8  # hard reserve / safety margin
    hover_power_factor: float = 1.0

    def __post_init__(self):
        self.hover_power_w = (
            self.battery_wh / (self.hover_endurance_min / 60.0)
        ) * self.hover_power_factor
        self.cruise_power_w = self.hover_power_w * self.cruise_power_factor
        self.usable_energy_wh = self.battery_wh * self.usable_battery_fraction

    # ------------------------------------------------------------------ #
    def ground_speed_mps(self, headwind_mps: float) -> float:
        """Ground speed for a given (signed) along-track headwind.

        Positive ``headwind_mps`` opposes motion (slows the drone), negative is
        a tailwind.  The drone holds a constant *airspeed*, so ground speed =
        airspeed - headwind, clamped to a sane band.
        """
        v = self.cruise_airspeed_mps - float(headwind_mps)
        return float(min(self.max_speed_mps, max(self.min_speed_mps, v)))

    def leg_duration_s(self, distance_m: float, headwind_mps: float) -> float:
        return float(distance_m) / self.ground_speed_mps(headwind_mps)

    def leg_energy_wh(self, distance_m: float, payload_kg: float, headwind_mps: float) -> float:
        """Linear battery drain for a cruise leg of ``distance_m`` metres.

        drain = P_cruise * flight_time * (1 + k_p*payload) * (1 + k_w*hw/v_air)

        Flight time already grows as ground speed drops into a headwind; the
        ``wind_energy_coef`` term additionally accounts for the extra thrust
        needed to make progress into the wind.  Linear & documented, per the
        demo requirement (not certified telemetry).
        """
        dur_h = self.leg_duration_s(distance_m, headwind_mps) / 3600.0
        payload_term = 1.0 + self.payload_energy_per_kg * max(0.0, payload_kg)
        wind_frac = max(0.0, headwind_mps) / max(1e-6, self.cruise_airspeed_mps)
        wind_term = 1.0 + self.wind_energy_coef * wind_frac
        return self.cruise_power_w * dur_h * payload_term * wind_term

    def hover_energy_wh(self, duration_s: float, payload_kg: float) -> float:
        """Energy burned loitering / hovering (drops, turns) under load."""
        payload_term = 1.0 + self.payload_energy_per_kg * max(0.0, payload_kg)
        return self.hover_power_w * (float(duration_s) / 3600.0) * payload_term

    def battery_fraction(self, energy_wh: float) -> float:
        if self.battery_wh <= 0:
            return 1.0
        return float(energy_wh) / self.battery_wh


# --------------------------------------------------------------------------- #
# Wind helpers (image coordinates: +x = East, +y = South, as used elsewhere)
# --------------------------------------------------------------------------- #
def wind_to_unit(from_deg: float) -> tuple:
    """Unit vector a wind blowing *from* ``from_deg`` travels *toward*.

    Meteorological 0 deg = from North (blowing South, +y in image coords).
    """
    r = math.radians(float(from_deg))
    # From-north wind moves toward +y (south): (sin, -cos) with +180 applied.
    r_to = r + math.pi
    return (math.sin(r_to), -math.cos(r_to))


def headwind_component_mps(dx: float, dy: float, weather: Dict[str, float]) -> float:
    """Along-track headwind (m/s) for travel vector (dx, dy).

    Positive = wind opposing motion; negative = tailwind.
    """
    wind_speed = float(weather.get("wind_speed_10m", 0.0))
    if wind_speed <= 0:
        return 0.0
    norm = math.hypot(dx, dy)
    if norm < 1e-9:
        return 0.0
    from_deg = float(weather.get("wind_direction_10m", 0.0))
    # Unit vector pointing *into* the wind (toward its origin) == travel dir for
    # which the wind is a pure headwind.
    r = math.radians(from_deg)
    into_wind = (math.sin(r), -math.cos(r))
    dot = (dx * into_wind[0] + dy * into_wind[1]) / norm
    return float(wind_speed * dot)


def estimate_straight_leg(
    start_px, goal_px, meters_per_pixel: float, spec: DroneSpec,
    weather: Dict[str, float], payload_kg: float = 0.0,
) -> dict:
    """Straight-line leg estimate (no obstacle detour) used for feasibility
    gates before an expensive D* Lite search."""
    dx = float(goal_px[0] - start_px[0])
    dy = float(goal_px[1] - start_px[1])
    dist_m = math.hypot(dx, dy) * meters_per_pixel
    hw = headwind_component_mps(dx, dy, weather)
    dur_s = spec.leg_duration_s(dist_m, hw)
    return {
        "distance_m": dist_m,
        "headwind_mps": hw,
        "ground_speed_mps": spec.ground_speed_mps(hw),
        "duration_s": dur_s,
        "energy_wh": spec.leg_energy_wh(dist_m, payload_kg, hw),
    }
