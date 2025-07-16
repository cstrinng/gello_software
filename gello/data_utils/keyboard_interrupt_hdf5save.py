import datetime
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Any, Dict, List

import numpy as np
import tyro
import h5py

from gello.agents.agent import BimanualAgent, DummyAgent
from gello.agents.gello_agent import GelloAgent
from gello.data_utils.format_obs import save_frame
from gello.env import RobotEnv
from gello.robots.robot import PrintRobot
from gello.zmq_core.robot_node import ZMQClientRobot
from gello.data_utils.keyboard_interface import KBReset


def main(args):
    # ... existing initialization ...

    # Initialize keyboard interface with optional frame cutoff
    kb_interface = KBReset(cut_frames=args.cut_frames, frames=args.frames)

    print("Start control loop...")

    save_path: Optional[Path] = None
    recording: bool = False
    buffer: List[Tuple[Dict[str, Any], np.ndarray]] = []

    obs = env.get_obs()
    start_time = time.time()
    while True:
        # Compute action
        action = agent.act(obs)

        # Update keyboard state
        state = kb_interface.update()
        if state == "start":
            # Begin recording: make folder and clear buffer
            dt_time = datetime.datetime.now()
            save_path = (
                Path(args.data_dir).expanduser() /
                args.agent /
                dt_time.strftime("%m%d_%H%M%S")
            )
            save_path.mkdir(parents=True, exist_ok=True)
            print(f"Recording to {save_path}")
            buffer.clear()
            recording = True

        elif state == "save" and recording:
            # Append one timestep to buffer
            buffer.append((obs.copy(), action.copy()))

        elif state == "normal" and recording:
            # End of recording: flush buffer to HDF5
            if save_path and buffer:
                # Create HDF5 file
                h5_file = save_path / "data.hdf5"
                with h5py.File(h5_file, 'w') as f:
                    grp = f.create_group('data')
                    # keys from first obs
                    keys = list(buffer[0][0].keys())
                    # prepare arrays
                    for key in keys:
                        # stack obs values
                        data_arr = np.stack([b[0][key] for b in buffer], axis=0)
                        grp.create_dataset(key, data=data_arr)
                    # actions
                    acts = np.stack([b[1] for b in buffer], axis=0)
                    grp.create_dataset('actions', data=acts)
                print(f"Saved HDF5: {h5_file} ({len(buffer)} frames)")
            # reset
            recording = False
            buffer.clear()
            save_path = None

        # Step environment
        obs = env.step(action)
        dt = time.time() - start_time
        print(f"Time step: {dt:.3f}s")


@dataclass
class Args:
    agent: str = "none"
    # ... other args ...
    use_save_interface: bool = False
    cut_frames: bool = False
    frames: Optional[int] = None  # no default cutoff if None
    data_dir: str = "~/bc_data"

if __name__ == "__main__":
    main(tyro.cli(Args))
