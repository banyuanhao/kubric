# Copyright 2024 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import numpy as np
import kubric as kb
from kubric.renderer.blender import Blender as KubricBlender
from kubric.simulator.pybullet import PyBullet as KubricSimulator
import cv2

def write_video(data, output_path, fps=24):
    # Ensure data is in the correct shape
    assert data.ndim == 4, "Data should be a 4D numpy array"
    assert data.shape[3] == 4, "The fourth dimension should be 4 (RGBA)"
    
    # Get the dimensions of the frames
    num_frames, height, width, channels = data.shape
    assert channels == 4, "The fourth dimension should be 4 (RGBA)"

    # Convert RGBA to RGB (discard the alpha channel)
    data_rgb = data[:, :, :, :3]
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 output
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Ensure frame is in the correct type
        frame = data_rgb[i].astype(np.uint8)
        out.write(frame)

    out.release()


logging.basicConfig(level="INFO")  # < CRITICAL, ERROR, WARNING, INFO, DEBUG

# --- create scene and attach a renderer and simulator
scene = kb.Scene(resolution=(256, 256))
scene.frame_end = 48   # < numbers of frames to render
scene.frame_rate = 24  # < rendering framerate
scene.step_rate = 240  # < simulation framerate
renderer = KubricBlender(scene)
simulator = KubricSimulator(scene)

# --- populate the scene with objects, lights, cameras
scene += kb.Cube(name="floor", scale=(3, 3, 0.1), position=(0, 0, -0.1),
                 static=True)
scene += kb.DirectionalLight(name="sun", position=(-1, -0.5, 3),
                             look_at=(0, 0, 0), intensity=1.5)
scene.camera = kb.PerspectiveCamera(name="camera", position=(2, -0.5, 4),
                                    look_at=(0, 0, 0))

# --- generates spheres randomly within a spawn region
spawn_region = [[-2, -1, 0], [-1.9, 1, 0]]
rng = np.random.default_rng()
for i in range(4):
  velocity = rng.uniform([0.5, -1, 0], [2.5, 1, 0])
  material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng))
  sphere = kb.Sphere(scale=0.1, velocity=velocity, material=material)
  scene += sphere
  kb.move_until_no_overlap(sphere, simulator, spawn_region=spawn_region)

# --- executes the simulation (and store keyframes)
simulator.run()

# --- renders the output
# renderer.save_state("output/simulator.blend")
frames_dict = renderer.render()
kb.write_image_dict({"rgba":frames_dict["rgba"]}, "output")
write_video(frames_dict["rgba"], "output/simulator_1.mp4")
