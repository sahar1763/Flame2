import numpy as np
from scipy.spatial.transform import Rotation as R


def camera_angle_from_vertical(
        platform_roll_deg: float,
        platform_pitch_deg: float,
        platform_yaw_deg: float,
        sensor_azimuth_deg: float,
        sensor_elevation_deg: float,
) -> float:
    """
    Compute the angle between the camera main optical axis and the vertical
    direction toward the ground (Down / Nadir).

    Coordinate system:
    - World frame: NED (North-East-Down), right-handed.
      X = North (Forward), Y = East (Right), Z = Down.

    - Camera canonical viewing direction (at 0,0,0):
      Points along the X-axis (Forward).

    Parameters
    ----------
    platform_roll_deg : float
        Platform roll (Tag 7). Positive = Right wing down.
    platform_pitch_deg : float
        Platform pitch (Tag 6). Positive = Nose up.
    platform_yaw_deg : float
        Platform heading (Tag 5). 0=North, 90=East.
    sensor_azimuth_deg : float
        Sensor relative yaw (Tag 18). 0=Forward, Positive=Clockwise.
    sensor_elevation_deg : float
        Sensor relative elevation (Tag 19). Negative=Down.

    Returns
    -------
    float
        Angle in degrees between the camera viewing axis and the vertical
        direction toward the ground (Down / +Z).
        0.0 = Camera looking straight down.
    """

    # --- 1. Platform Rotation ---
    # "Start with Yaw, then Pitch, then Roll."
    # Axes mapping: Yaw=Z, Pitch=Y, Roll=X.
    # Scipy convention 'zyx' corresponds to Intrinsic Z -> Y -> X.
    platform2world = R.from_euler(
        'ZYX', # TODO: Make sure intrinsic rotations or extrinsic rotations
        [platform_yaw_deg, platform_pitch_deg, platform_roll_deg],
        degrees=True
    ).as_matrix()

    # --- 2. Sensor Rotation ---
    # Sensor rotates relative to the platform body.
    # Order: Azimuth (Yaw around Z) -> Elevation (Pitch around Y).
    # Negative elevation is down.
    # (Scipy 'y' rotation of -90 on vector [1,0,0] results in [0,0,1] which is Down/+Z,
    # so the sign convention is handled correctly naturally).
    sensor2platform = R.from_euler(
        'ZY', # TODO: Make sure intrinsic rotations or extrinsic rotations
        [sensor_azimuth_deg, sensor_elevation_deg],
        degrees=True
    ).as_matrix()

    # --- 3. Canonical Camera Vector ---
    # "0 degrees forward along the longitudinal axis".
    # In NED, Longitudinal/Forward is +X.
    v_camera_canonical = np.array([1.0, 0.0, 0.0])

    # --- 4. Calculate World Vector ---
    # Apply sensor rotation, then platform rotation
    v_sensor_body = sensor2platform @ v_camera_canonical
    v_sensor_world = platform2world @ v_sensor_body

    # --- 5. Define Down Vector ---
    # In NED, Down is +Z
    v_down = np.array([0.0, 0.0, 1.0])

    # --- 6. Calculate Angle ---
    # Dot product: a . b = |a||b| cos(theta)
    # Since vectors are normalized (length 1), cos(theta) = a . b
    dot_prod = np.dot(v_sensor_world, v_down)

    # Clip for numerical stability
    cos_theta = np.clip(dot_prod, -1.0, 1.0)

    angle_rad = np.arccos(cos_theta)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


if __name__ == "__main__":

    platform_roll_deg = 90
    platform_pitch_deg = 0
    platform_yaw_deg = 0
    sensor_azimuth_deg = 90
    sensor_elevation_deg = 0

    angle = camera_angle_from_vertical(
        platform_roll_deg=platform_roll_deg,
        platform_pitch_deg=platform_pitch_deg,
        platform_yaw_deg=platform_yaw_deg,
        sensor_azimuth_deg=sensor_azimuth_deg,
        sensor_elevation_deg=sensor_elevation_deg
    )
    print(f"Camera angle from vertical: {angle:.2f} degrees")
