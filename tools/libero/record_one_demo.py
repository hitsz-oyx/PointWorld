"""Record a single synthetic LIBERO demo and save it as HDF5.

The output HDF5 matches the schema that ``tools.libero.export_clip``
expects, i.e. it has a ``data/demo_0`` group with ``actions`` and
``states`` datasets, plus the ``model_file`` and ``env_args`` attributes
that the exporter reads to rebuild the env.

This is meant for smoke-testing the clip-exporter path end-to-end
without downloading any of the official LIBERO dataset tarballs. The
"actions" played back are random end-effector deltas in the OSC_POSE
action space; we are only interested in producing a demo HDF5 with the
right structure, not in solving the task.

Usage (run inside the LIBERO conda env)::

    python -m tools.libero.record_one_demo \
        --bddl /path/to/some/libero_task.bddl \
        --output /tmp/libero_demo.hdf5 \
        --num_steps 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

# Locate the LIBERO source checkout so we can reuse its scripts/init_path
# and its BDDL problem parser. The env variable is the only configuration
# the user needs to provide; the default matches the env we just set up.
import os as _os
_LIBERO_SRC = Path(
    _os.environ.get(
        "LIBERO_SRC",
        Path(__file__).resolve().parents[3] / "LIBERO",
    )
)
if not _LIBERO_SRC.is_dir():
    raise FileNotFoundError(
        f"Could not find LIBERO source at {_LIBERO_SRC}. "
        "Either clone https://github.com/Lifelong-Robot-Learning/LIBERO "
        "there or set the LIBERO_SRC env var."
)
sys.path.insert(0, str(_LIBERO_SRC))
sys.path.insert(0, str(_LIBERO_SRC / "scripts"))
import init_path  # noqa: F401  (adds scripts/ to sys.path inside LIBERO)
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import TASK_MAPPING


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bddl",
        required=True,
        help="Path to a LIBERO BDDL task file (must be reachable from the env).",
    )
    p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Destination HDF5 file (will be created or overwritten).",
    )
    p.add_argument(
        "--num_steps",
        type=int,
        default=60,
        help="Number of (state, action) pairs to record. >= 50 is recommended "
        "so the clip exporter has enough room to pick a start_idx.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def record_demo(bddl_file: str, num_steps: int, seed: int) -> dict:
    """Build a LIBERO env for the given BDDL, run ``num_steps`` random
    OSC_POSE actions, and return the recorded trajectory together with
    the env metadata (model_file, env_args) needed to recreate the env.
    """
    problem_info = BDDLUtils.get_problem_info(bddl_file)
    problem_name = problem_info["problem_name"]
    print(f"[record] task = {problem_name}", file=sys.stderr)
    print(f"[record] instruction = {problem_info['language_instruction']}", file=sys.stderr)

    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file,
        robots=["Panda"],
        controller_configs={
            "type": "OSC_POSE",
            "input_max": 1,
            "input_min": -1,
            "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
            "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
            "kp": 150,
            "damping_ratio": 1,
            "impedance_mode": "fixed",
            "kp_limits": [0, 300],
            "damping_ratio_limits": [0, 10],
            "position_limits": None,
            "orientation_limits": None,
            "uncouple_pos_ori": True,
            "control_delta": True,
            "interpolation": None,
            "ramp_ratio": 0.2,
        },
        has_renderer=False,
        has_offscreen_renderer=False,
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=False,
        control_freq=20,
    )

    rng = np.random.default_rng(seed)
    actions = []
    states = []
    env.reset()
    # Capture the initial frame's sim state so demo_0 starts at env.reset().
    states.append(np.array(env.sim.get_state().flatten(), dtype=np.float32))
    for step in range(num_steps):
        # Random 6-DoF delta pose + gripper (1 = open, -1 = close) = 7-D.
        action = np.zeros(7, dtype=np.float32)
        action[:6] = rng.uniform(-0.3, 0.3, size=6).astype(np.float32)
        action[6] = float(rng.choice([-1.0, 1.0]))
        env.step(action)
        actions.append(action.astype(np.float32))
        states.append(np.array(env.sim.get_state().flatten(), dtype=np.float32))

    # model_file (full MuJoCo XML string) — exporter postprocesses this
    # to resolve mesh paths. Use the env's current sim XML.
    model_file = env.sim.model.get_xml()

    env_args = {
        "bddl_file_name": str(Path(bddl_file).resolve()),
        "robots": ["Panda"],
    }

    return {
        "actions": np.stack(actions, axis=0).astype(np.float32),
        "states": np.stack(states, axis=0).astype(np.float32),
        "model_file": model_file,
        "env_args": env_args,
        "problem_name": problem_name,
    }


def save_demo(payload: dict, output_path: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        data_grp = f.create_group("data")
        demo_grp = data_grp.create_group("demo_0")
        demo_grp.create_dataset(
            "actions", data=payload["actions"], compression="gzip"
        )
        demo_grp.create_dataset(
            "states", data=payload["states"], compression="gzip"
        )
        # The MuJoCo XML for a LIBERO scene is large (>64KB) and would
        # overflow HDF5's per-attribute size limit. Store it as a scalar
        # dataset of ``bytes``; ``tools.libero.export_clip`` reads
        # ``demo_group.attrs["model_file"]`` if it's small but
        # automatically falls back to this dataset if the attr is missing.
        # We populate the attribute first (so the happy path is unchanged)
        # and only fall back to the dataset when the XML exceeds 60 KB.
        xml_bytes = payload["model_file"].encode("utf-8")
        if len(xml_bytes) < 60_000:
            demo_grp.attrs["model_file"] = np.bytes_(xml_bytes)
        else:
            demo_grp.create_dataset(
                "model_file",
                data=np.bytes_(xml_bytes),
            )
        # store env_args as a JSON string for round-trip safety
        demo_grp.attrs["env_args"] = json.dumps(payload["env_args"])
        demo_grp.attrs["num_samples"] = int(payload["actions"].shape[0])
        demo_grp.attrs["problem_name"] = payload["problem_name"]
    print(f"[record] wrote {output_path}", file=sys.stderr)


def main() -> int:
    args = _parse_args()
    payload = record_demo(args.bddl, args.num_steps, args.seed)
    save_demo(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
