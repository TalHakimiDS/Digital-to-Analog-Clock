"""
Utility functions for time manipulation and conversion
"""

import numpy as np
import random
from typing import Tuple, List


def generate_unique_times(num_times: int = 400, seed: int = 42) -> List[Tuple[int, int, int]]:
    """
    Generate unique (hour, minute, second) combinations.

    Args:
        num_times: Number of unique time combinations (max 400 as per requirements)
        seed: Random seed for reproducibility

    Returns:
        List of (hour, minute, second) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    times = set()

    # Strategy: Sample uniformly across the day
    # Total seconds in a day: 24 * 60 * 60 = 86400
    # We need 400 samples, so roughly every 216 seconds

    if num_times <= 400:
        # Sample specific times to ensure good coverage
        for h in range(24):
            for m in range(0, 60, max(1, 60 // (num_times // 24 + 1))):
                for s in range(0, 60, max(1, 60 // (num_times // 24 + 1))):
                    times.add((h, m, s))
                    if len(times) >= num_times:
                        break
                if len(times) >= num_times:
                    break
            if len(times) >= num_times:
                break

    times_list = list(times)[:num_times]

    # Shuffle to randomize
    random.shuffle(times_list)

    print(f"✓ Generated {len(times_list)} unique time combinations")
    return times_list


def time_to_angles(hour: int, minute: int, second: int) -> Tuple[float, float, float]:
    """
    Convert time to hand angles in degrees.

    Args:
        hour: Hour (0-23)
        minute: Minute (0-59)
        second: Second (0-59)

    Returns:
        Tuple of (hour_angle, minute_angle, second_angle) in degrees [0, 360)
    """
    # Second hand: 6 degrees per second
    second_angle = second * 6.0

    # Minute hand: 6 degrees per minute + 0.1 degrees per second
    minute_angle = minute * 6.0 + second * 0.1

    # Hour hand: 30 degrees per hour + 0.5 degrees per minute + 0.00833 degrees per second
    hour_12 = hour % 12  # Convert to 12-hour format
    hour_angle = hour_12 * 30.0 + minute * 0.5 + second * (0.5 / 60.0)

    return hour_angle, minute_angle, second_angle


def angles_to_sincos(hour_angle: float, minute_angle: float, second_angle: float) -> np.ndarray:
    """
    Convert angles to sin/cos representation to handle angle wraparound.

    Args:
        hour_angle, minute_angle, second_angle: Angles in degrees

    Returns:
        numpy array of shape (6,): [sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]
    """
    hour_rad = np.deg2rad(hour_angle)
    minute_rad = np.deg2rad(minute_angle)
    second_rad = np.deg2rad(second_angle)

    return np.array([
        np.sin(hour_rad), np.cos(hour_rad),
        np.sin(minute_rad), np.cos(minute_rad),
        np.sin(second_rad), np.cos(second_rad)
    ], dtype=np.float32)


def sincos_to_angles(sincos: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert sin/cos representation back to angles.

    Args:
        sincos: Array of shape (6,): [sin_h, cos_h, sin_m, cos_m, sin_s, cos_s]

    Returns:
        Tuple of (hour_angle, minute_angle, second_angle) in degrees [0, 360)
    """
    sin_h, cos_h, sin_m, cos_m, sin_s, cos_s = sincos

    hour_angle = np.rad2deg(np.arctan2(sin_h, cos_h)) % 360
    minute_angle = np.rad2deg(np.arctan2(sin_m, cos_m)) % 360
    second_angle = np.rad2deg(np.arctan2(sin_s, cos_s)) % 360

    return hour_angle, minute_angle, second_angle


def angles_to_time(hour_angle: float, minute_angle: float, second_angle: float) -> Tuple[int, int, int]:
    """
    Convert hand angles back to time (approximate).

    Args:
        hour_angle, minute_angle, second_angle: Angles in degrees

    Returns:
        Tuple of (hour, minute, second)
    """
    # Second from second hand
    second = int(round(second_angle / 6.0)) % 60

    # Minute from minute hand
    minute = int(round(minute_angle / 6.0)) % 60

    # Hour from hour hand
    hour = int(round(hour_angle / 30.0)) % 12

    return hour, minute, second


def normalize_time(hour: int, minute: int, second: int) -> np.ndarray:
    """
    Normalize time values to [0, 1] range for regression.

    Args:
        hour, minute, second: Time components

    Returns:
        numpy array of shape (3,): normalized [hour, minute, second]
    """
    return np.array([
        hour / 23.0,
        minute / 59.0,
        second / 59.0
    ], dtype=np.float32)


def denormalize_time(normalized: np.ndarray) -> Tuple[int, int, int]:
    """
    Denormalize time values from [0, 1] to actual time.

    Args:
        normalized: Array of shape (3,): normalized [hour, minute, second]

    Returns:
        Tuple of (hour, minute, second)
    """
    hour = int(round(normalized[0] * 23.0))
    minute = int(round(normalized[1] * 59.0))
    second = int(round(normalized[2] * 59.0))

    return hour, minute, second


def time_to_seconds(hour: int, minute: int, second: int) -> int:
    """Convert time to total seconds since midnight."""
    return hour * 3600 + minute * 60 + second


def seconds_to_time(seconds: int) -> Tuple[int, int, int]:
    """Convert total seconds to (hour, minute, second)."""
    hour = (seconds // 3600) % 24
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return hour, minute, second


def calculate_time_difference(time1: Tuple[int, int, int], time2: Tuple[int, int, int]) -> int:
    """
    Calculate absolute difference between two times in seconds.

    Args:
        time1, time2: (hour, minute, second) tuples

    Returns:
        Difference in seconds
    """
    s1 = time_to_seconds(*time1)
    s2 = time_to_seconds(*time2)
    return abs(s1 - s2)


def calculate_angle_error(pred_angle: float, true_angle: float) -> float:
    """
    Calculate the smallest angle difference (handles wraparound).

    Args:
        pred_angle, true_angle: Angles in degrees

    Returns:
        Minimum angle difference in degrees [0, 180]
    """
    diff = abs(pred_angle - true_angle) % 360
    return min(diff, 360 - diff)
