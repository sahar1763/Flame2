import numpy as np
from scipy.spatial.transform import Rotation as R


def camera_angle_from_down_NED(
    platform_roll_deg: float,
    platform_pitch_deg: float,
    platform_yaw_deg: float,
    sensor_azimuth_deg: float,
    sensor_elevation_deg: float,
) -> float:
    """
    Returns angle (deg) between camera LOS and NED Down (+Z).

    Output meaning:
      0°   -> looking straight down (nadir)
      90°  -> horizontal
      180° -> looking straight up

    Inputs meaning (per your description + MISB style):
      sensor_elevation_deg = 0° is horizontal
      sensor_elevation_deg < 0° points down
      sensor_elevation_deg > 0° points up
    """

    # Platform: intrinsic Yaw -> Pitch -> Roll
    R_body_to_world = R.from_euler(
        "zyx",
        [platform_yaw_deg, platform_pitch_deg, platform_roll_deg],
        degrees=True
    ).as_matrix()

    # Sensor: intrinsic Azimuth -> Elevation -> Roll
    R_sensor_to_body = R.from_euler(
        "ZY",
        [sensor_azimuth_deg, sensor_elevation_deg],
        degrees=True
    ).as_matrix()

    # Camera LOS in sensor frame: forward along +X
    v_cam_sensor = np.array([1.0, 0.0, 0.0])

    # Transform LOS to world (NED)
    v_cam_world = R_body_to_world@R_sensor_to_body@v_cam_sensor
    # v_cam_world=v_cam_sensor
    v_cam_world = v_cam_world / np.linalg.norm(v_cam_world)

    # NED Down axis
    v_down = np.array([0.0, 0.0, 1.0])

    # Angle from Down
    cos_theta = np.clip(np.dot(v_cam_world, v_down), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


if __name__ == "__main__":
    # Level drone, yaw irrelevant to "down" angle:
    angle= camera_angle_from_down_NED(
    platform_roll_deg  = 90,
    platform_pitch_deg = 40,
    platform_yaw_deg =40,
    sensor_azimuth_deg = 40,
    sensor_elevation_deg = 50,
    )

    print(f"{angle:.2f}")


