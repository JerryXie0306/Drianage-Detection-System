from dataclasses import dataclass


@dataclass
class DetectorConfig:
    # Region of interest as fractions (x0, y0, x1, y1) of the frame: the grate / inlet area.
    roi: tuple = (0.0, 0.0, 1.0, 1.0)
    max_width: int = 320
    # Debris coverage (fraction of ROI) boundaries for the heuristic backend.
    clear_below: float = 0.08
    blocked_above: float = 0.30
    min_component_frac: float = 0.003
    # segment backend: % of the grate region covered by debris
    seg_clear_below: float = 10.0
    seg_blocked_above: float = 40.0


@dataclass
class ControllerConfig:
    confidence_threshold: float = 0.80  # matches the 80% threshold in the app wireframe
    open_frames: int = 3  # consecutive blocked frames before auto-open (debounce)
    close_frames: int = 10  # consecutive clear frames before auto-close
    min_open_seconds: float = 30.0  # cooldown so the cover does not chatter
    confirm_timeout: float = 10.0  # no confirmation from actuator -> fault
    camera_online_s: float = 15.0  # a frame older than this shows the camera as offline (raise for slow schedules)
    open_on_partial: bool = False


@dataclass
class ActuatorConfig:
    delay_s: float = 2.0  # simulated open/close time
    jitter_s: float = 0.3
    fail_rate: float = 0.0  # simulated command failure probability (for fault testing)
