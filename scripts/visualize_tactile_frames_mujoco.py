from __future__ import annotations

import argparse
from pathlib import Path

import glfw
import mujoco
import numpy as np
import pandas as pd

DEFAULT_CSV = Path(r"d:\datasets\tac2Slip\severity-04-23\only_translation\tactile_data0000.csv")
FINGERS = ["ff", "mf", "rf", "lf", "th"]
POSE_SUFFIXES = ["x", "y", "z", "qx", "qy", "qz", "qw"]
OBJECT_POSE_COLUMNS = ["px", "py", "pz", "ox", "oy", "oz", "ow"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse tactile frames in MuJoCo with keyboard control.")
    parser.add_argument("csv_path", nargs="?", default=str(DEFAULT_CSV), help="Path to tactile CSV file")
    parser.add_argument("--start-row", type=int, default=0, help="Initial row index")
    parser.add_argument("--width", type=int, default=1400, help="Window width")
    parser.add_argument("--height", type=int, default=900, help="Window height")
    return parser.parse_args()


def load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = []
    for finger in FINGERS:
        required.extend([f"{finger}_{suffix}" for suffix in POSE_SUFFIXES])
    required.extend(OBJECT_POSE_COLUMNS)
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    return df


def build_model_xml() -> str:
    finger_bodies = []
    colors = {
        "x": "1 0.1 0.1 1",
        "y": "0.1 1 0.1 1",
        "z": "0.1 0.3 1 1",
    }
    for finger in FINGERS:
        finger_bodies.append(
            f"""
            <body name="{finger}_frame" mocap="true" pos="0 0 0" quat="1 0 0 0">
              <geom type="capsule" fromto="0 0 0 0.02 0 0" size="0.002" rgba="{colors['x']}"/>
              <geom type="capsule" fromto="0 0 0 0 0.02 0" size="0.002" rgba="{colors['y']}"/>
              <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.002" rgba="{colors['z']}"/>
            </body>
            """
        )

    finger_xml = "\n".join(finger_bodies)

    return f"""
    <mujoco model="tactile_frames">
      <compiler angle="radian" coordinate="local"/>
      <option gravity="0 0 0" integrator="Euler" timestep="0.01"/>
      <visual>
        <map znear="0.01" zfar="10"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.9 0.95 1" rgb2="0.2 0.2 0.25" width="256" height="256"/>
        <material name="floor" rgba="0.92 0.92 0.95 1" specular="0.1" shininess="0.1"/>
        <material name="box" rgba="0.75 0.55 0.25 0.45" specular="0.2" shininess="0.1"/>
      </asset>
      <worldbody>
        <light pos="0 0 2" dir="0 0 -1" diffuse="0.7 0.7 0.7" specular="0.2 0.2 0.2"/>
        <geom type="plane" size="2 2 0.01" material="floor"/>
        {finger_xml}
        <body name="object_frame" mocap="true" pos="0 0 -0.05" quat="1 0 0 0">
                    <geom type="capsule" fromto="0 0 0 0.02 0 0" size="0.0025" rgba="1 0.1 0.1 1"/>
                    <geom type="capsule" fromto="0 0 0 0 0.02 0" size="0.0025" rgba="0.1 1 0.1 1"/>
                    <geom type="capsule" fromto="0 0 0 0 0 0.02" size="0.0025" rgba="0.1 0.3 1 1"/>
          <geom name="object_box" type="box" size="0.10 0.10 0.008" material="box"/>
        </body>
      </worldbody>
    </mujoco>
    """


def quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def set_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, pos: np.ndarray, quat_xyzw: np.ndarray) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    mocap_id = model.body_mocapid[body_id]
    if mocap_id < 0:
        raise ValueError(f"Body {body_name} is not mocap-enabled")
    data.mocap_pos[mocap_id] = pos
    data.mocap_quat[mocap_id] = quat_xyzw_to_wxyz(quat_xyzw)


def apply_row(df: pd.DataFrame, model: mujoco.MjModel, data: mujoco.MjData, row_idx: int) -> tuple[np.ndarray, np.ndarray]:
    row = df.iloc[row_idx]

    for finger in FINGERS:
        pos = row[[f"{finger}_x", f"{finger}_y", f"{finger}_z"]].to_numpy(dtype=np.float32)
        quat = row[[f"{finger}_qx", f"{finger}_qy", f"{finger}_qz", f"{finger}_qw"]].to_numpy(dtype=np.float32)
        set_body_pose(model, data, f"{finger}_frame", pos, quat)

    object_pos = row[["px", "py", "pz"]].to_numpy(dtype=np.float32)
    object_quat = row[["ox", "oy", "oz", "ow"]].to_numpy(dtype=np.float32)
    set_body_pose(model, data, "object_frame", object_pos, object_quat)

    mujoco.mj_forward(model, data)
    return object_pos, object_quat


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv_path)
    csv_path = r"D:\datasets\tac2Slip\severity-04-23\unconstrained\tactile_data0000.csv"
    df = load_csv(csv_path)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    try:
        xml = build_model_xml()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)

        window = glfw.create_window(args.width, args.height, "Tactile MuJoCo Frame Browser", None, None)
        if not window:
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.azimuth = 135
        cam.elevation = -20
        cam.distance = 0.45
        cam.lookat[:] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        scene = mujoco.MjvScene(model, maxgeom=2000)
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
        viewport = mujoco.MjrRect(0, 0, args.width, args.height)

        state = {"row": max(0, min(args.start_row, len(df) - 1))}
        state["autoplay"] = False
        state["last_step_time"] = glfw.get_time()
        autoplay_period = 0.15
        overlay_text = [
            "Right/Space: next row",
            "Left/Backspace: previous row",
            "A: toggle auto-play",
            "R: reset",
            "Esc: quit",
        ]

        def update_scene() -> None:
            row_idx = state["row"]
            object_pos, _ = apply_row(df, model, data, row_idx)
            cam.lookat[:] = object_pos

        def on_key(_window, key, _scancode, action, _mods):
            if action != glfw.PRESS:
                return
            if key in (glfw.KEY_RIGHT, glfw.KEY_SPACE, glfw.KEY_ENTER):
                state["row"] = min(state["row"] + 1, len(df) - 1)
                state["last_step_time"] = glfw.get_time()
                update_scene()
            elif key in (glfw.KEY_LEFT, glfw.KEY_BACKSPACE):
                state["row"] = max(state["row"] - 1, 0)
                state["last_step_time"] = glfw.get_time()
                update_scene()
            elif key == glfw.KEY_A:
                state["autoplay"] = not state["autoplay"]
                state["last_step_time"] = glfw.get_time()
            elif key == glfw.KEY_R:
                state["row"] = 0
                state["last_step_time"] = glfw.get_time()
                update_scene()
            elif key == glfw.KEY_END:
                state["row"] = len(df) - 1
                state["last_step_time"] = glfw.get_time()
                update_scene()
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(_window, True)

        glfw.set_key_callback(window, on_key)
        update_scene()

        while not glfw.window_should_close(window):
            if state["autoplay"]:
                now = glfw.get_time()
                if now - state["last_step_time"] >= autoplay_period:
                    if state["row"] < len(df) - 1:
                        state["row"] += 1
                        state["last_step_time"] = now
                        update_scene()
                    else:
                        state["autoplay"] = False
            width, height = glfw.get_framebuffer_size(window)
            viewport.width = width
            viewport.height = height
            mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
            mujoco.mjr_render(viewport, scene, context)
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_150,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                f"Row: {state['row']} / {len(df) - 1} | Auto-play: {'on' if state['autoplay'] else 'off'}",
                " | ".join(overlay_text),
                context,
            )
            glfw.swap_buffers(window)
            glfw.poll_events()

    finally:
        glfw.terminate()


if __name__ == "__main__":
    main()
