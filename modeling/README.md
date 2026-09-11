# modular-origami-soft-robot-mujoco
A MuJoCo-based framework for soft robotic arm modeling, physics simulation, and grasping experiments with contact-force analysis.

## Overview

The goal of this project is to build a simulation pipeline for a soft robotic arm, enabling:

- **Soft robotic arm modeling**
  - Build a multi-segment soft/continuum-like robotic arm model in MuJoCo.
- **Physics simulation**
  - Simulate deformation-related motion, actuation, and object interaction using MuJoCo dynamics.
- **Grasping task validation**
  - Test whether the robotic arm can approach, contact, and grasp target objects.
- **Actuator control**
  - Support rope/tendon-like actuation for driving arm motion.
- **Contact force analysis**
  - Extract and visualize contact forces during object interaction and grasping.

## Project Structure

```text
.
├── xml/                    # XML templates and generated MuJoCo models
├── py/                     # intermediate scripts for model generation/assembly
├── STL/                     # STL mesh files for the robot and grasped objects (not included in this repository)
├── merge.py                # model generation and XML assembly pipeline
├── compute.py            # simulation, control, and contact-force visualization
└── README.md
```
## Usage

### Generate a full model

```bash
python merge.py --start 1 --end 20 --max-force 10
```

### Arguments

- `--start`: top module index
- `--end`: bottom module index
- `--max-force`: actuator control upper bound

This will generate a final assembled XML model such as:

```text
final1_20.xml
```

### Simulation

Run the MuJoCo simulation and visualize contact forces:

```bash
python compute.py
```

The simulation script supports:

- rope actuator control
- tracked contact force extraction
- per-body contact force logging
- contact force arrow rendering in the MuJoCo viewer

## Dependencies

- Python 3.x
- MuJoCo
- NumPy

## Simulation Result

<p>
  The image shows the initial posture of the soft robotic arm in the simulation.  
</p>

<p align="center">
  <img src="images/scene.png" alt="Initial posture" width="70%">
</p>

<p>
  The image shows the robotic arm grasping the target object.
</p>

<p align="center">
  <img src="images/grasp_result.png" alt="Grasping result" width="70%">
</p>
