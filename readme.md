# DeepSlicing: Deep Reinforcement Learning for 5G Network Slicing

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)

## Problem Statement

5G networks introduce the concept of network slicing, which allows operators to create multiple virtual networks (slices) on a shared physical infrastructure. Each slice is tailored to serve specific applications with distinct Quality of Service (QoS) requirements:

- **eMBB (Enhanced Mobile Broadband)**: High data rates for applications like video streaming and web browsing
- **URLLC (Ultra-Reliable Low-Latency Communications)**: Low latency and high reliability for applications like autonomous vehicles and industrial automation
- **mMTC (Massive Machine Type Communications)**: Support for a large number of connected devices for IoT applications

The key challenge in network slicing is **dynamic resource allocation** - how to efficiently distribute limited network resources (bandwidth, computing power, etc.) among these slices while:

1. **Satisfying diverse QoS requirements** of each slice type
2. **Maximizing overall resource utilization**
3. **Adapting to time-varying traffic patterns**
4. **Enforcing resource constraints** (total available resources)

Traditional approaches like static allocation or simple heuristics fail to adapt to dynamic traffic patterns and often lead to either resource wastage or QoS violations.

## Solution Approach

DeepSlicing addresses this challenge by combining two powerful techniques:

1. **Multi-Agent Deep Deterministic Policy Gradient (MADDPG)**: A multi-agent reinforcement learning algorithm that enables each network slice to learn optimal resource allocation policies through interaction with the environment.

2. **Alternating Direction Method of Multipliers (ADMM)**: A mathematical optimization technique that decomposes the global resource allocation problem into smaller sub-problems while enforcing global resource constraints.

This hybrid approach allows us to:

- **Learn complex, non-linear relationships** between traffic patterns, QoS requirements, and optimal resource allocations
- **Adapt to dynamic traffic patterns** in real-time
- **Enforce strict resource constraints** while optimizing for multiple objectives
- **Scale to multiple slices** with different requirements

## Technical Innovations

DeepSlicing introduces several technical innovations:

1. **MADDPG-ADMM Integration**: A novel framework that combines the learning capabilities of MADDPG with the constraint handling of ADMM, enabling constrained reinforcement learning for network slicing.

2. **Centralized Training with Decentralized Execution**: Agents are trained with access to global information but can execute decisions using only local observations, making the system practical for real-world deployment.

3. **NS-3 Integration**: Realistic network simulation using NS-3, allowing the framework to be trained and evaluated in environments that closely model real 5G networks.

4. **Augmented Reward Function**: A carefully designed reward function that balances multiple objectives: QoS satisfaction, resource utilization, and constraint satisfaction.

5. **Production-Ready Deployment Pipeline**: Complete pipeline from training to deployment, including model export to various formats, containerization, Kubernetes deployment, and monitoring.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DeepSlicing Framework                          │
│                                                                         │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │ Network Slicing│    │  MADDPG Agent  │    │      ADMM Optimizer    │    │
│  │  Environment  │◄───┤   (Training)   │◄───┤   (Constraint Handling) │    │
│  │               │    │               │    │                       │    │
│  └───────┬───────┘    └───────┬───────┘    └───────────┬───────────┘    │
│          │                    │                        │                │
│          ▼                    ▼                        ▼                │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────────────┐    │
│  │   NS-3 Network │    │  Model Export  │    │  Monitoring & Metrics  │    │
│  │   Simulation  │    │   Pipeline    │    │  (Prometheus/Grafana)  │    │
│  │   (Optional)  │    │               │    │                       │    │
│  └───────────────┘    └───────┬───────┘    └───────────────────────┘    │
│                               │                                         │
│                               ▼                                         │
│                      ┌───────────────┐                                  │
│                      │  Deployment   │                                  │
│                      │  (Docker/K8s) │                                  │
│                      │               │                                  │
│                      └───────────────┘                                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Core Components

1. **MADDPG (Multi-Agent Deep Deterministic Policy Gradient)**
   - Implemented in `algorithms/maddpg_full.py`
   - Provides a centralized critic for training and decentralized actors for execution
   - Handles the learning process for optimal resource allocation policies

2. **ADMM Integration**
   - Implemented in `algorithms/admm_wrapper.py`
   - Decomposes the global resource allocation problem into sub-problems
   - Enforces global resource constraints while allowing distributed decision-making

3. **Network Slicing Environment**
   - Implemented in `gym_env/gym_env.py`
   - Models the dynamics of 5G network slicing
   - Provides state observations, processes actions, and calculates rewards
   - Optionally integrates with NS-3 for realistic network simulation

### Key Features

- **Training and Optimization**
  - Multi-agent reinforcement learning with MADDPG
  - Hyperparameter optimization using Optuna
  - NS-3 integration for realistic network simulation
  - Evaluation against baseline approaches

- **Evaluation**
  - Performance comparison with baseline approaches (Static, Proportional, Dynamic)
  - Metrics: QoS satisfaction, resource utilization, constraint violations
  - NS-3 integration for realistic evaluation

- **Deployment**
  - Model export to various formats (PyTorch, ONNX, TorchScript)
  - Flask API for model serving
  - Docker containerization
  - Kubernetes deployment
  - Prometheus/Grafana monitoring

## Project Structure

```
Deep Slicing Bundle/
├── algorithms/               # Core algorithms implementation
│   ├── admm_wrapper.py      # ADMM integration with DRL
│   └── maddpg_full.py       # MADDPG implementation
├── deploy/                  # Deployment configurations
│   ├── Dockerfile           # Container definition
│   ├── docker-compose.yml   # Multi-container setup
│   ├── kubernetes/          # K8s manifests
│   └── grafana/             # Monitoring dashboards
├── docs/                    # Documentation
│   ├── ns3_integration.md   # NS-3 integration guide
│   └── runbook.md           # Operations runbook
├── export/                  # Model export utilities
│   └── export_model.py      # Export script
├── gym_env/                 # OpenAI Gym environments
│   └── gym_env.py           # Network slicing environment
├── ns3/                     # NS-3 integration
│   └── ns3_sched_bridge.cc  # NS-3 bridge implementation
├── serve/                   # Model serving
│   └── app.py               # Flask API
├── evaluate.py              # Evaluation script
├── hyperparameter_optimization.py  # Hyperparameter tuning
├── requirements.txt         # Dependencies
├── train_maddpg.py          # Training script
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- OpenAI Gym
- (Optional) NS-3 for network simulation

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/deepslicing.git
cd deepslicing

# Install dependencies
pip install -r requirements.txt

# (Optional) Install NS-3 and ns3-gym
# See docs/ns3_integration.md for detailed instructions
```

## Usage

### Training

```bash
# Basic training with simulated environment
python train_maddpg.py --num_episodes 10000 --save_dir ./models

# Training with NS-3 integration
python train_maddpg.py --use_ns3 --num_episodes 5000 --save_dir ./models

# Hyperparameter optimization
python hyperparameter_optimization.py --n_trials 100 --study_name deepslicing
```

### Evaluation

```bash
# Evaluate trained model
python evaluate.py --model_path ./models/checkpoint_10000.pth

# Compare with baselines
python evaluate.py --model_path ./models/checkpoint_10000.pth --compare_baselines

# Evaluate with NS-3
python evaluate.py --model_path ./models/checkpoint_10000.pth --use_ns3
```

### Deployment

```bash
# Export model for deployment
python export/export_model.py --checkpoint ./models/checkpoint_10000.pth --format all

# Run API server
python serve/app.py --model_path ./models/checkpoint_10000.pth

# Deploy with Docker Compose
cd deploy
docker-compose up -d
```

See `deploy/deployment_guide.md` for detailed deployment instructions.

## Documentation

- `docs/runbook.md`: Comprehensive guide for operating the system
- `docs/ns3_integration.md`: Instructions for NS-3 integration
- `deploy/deployment_guide.md`: Deployment instructions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use DeepSlicing in your research, please cite our paper:

```
@article{deepslicing2023,
  title={DeepSlicing: Deep Reinforcement Learning for Resource Allocation in 5G Network Slicing},
  author={Your Name},
  journal={IEEE Journal on Selected Areas in Communications},
  year={2023},
  volume={},
  number={},
  pages={}
}
```

## Acknowledgments

- This project was inspired by research in network slicing optimization
- Thanks to the NS-3 community for their simulation framework
- PyTorch team for their deep learning library
