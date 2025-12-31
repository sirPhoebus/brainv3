# Omnidirectional Swarm System (AGI)

A modular AI architecture designed for reasoning through an omnidirectional swarm of agents. This system features a visual processing layer, an embedding language middleware, and a decentralized swarm core with curiosity-driven exploration and human-in-the-loop verification.

## Architecture

- **Visual Cortex (`src/cortex/`)**: Handles input segmentation and initial feature extraction. Includes a `MockCortex` for simulation.
- **Bridge (`src/bridge/`)**: Middleware that translates visual segments into prioritized `AgentTokens` using a standard embedding language.
- **Swarm Core (`src/swarm/`)**: 
    - **Omnidirectional Agents**: Reasoning entities that process tokens and propose hypotheses.
    - **Message Bus**: In-memory pub/sub for real-time agent communication and consensus.
    - **Verifier**: Independent quality control that prunes conflicting ideas.
- **Curiosity Module (`src/curiosity/`)**: Rewards novelty to ensure the swarm explores diverse reasoning paths.
- **HITL (`src/hitl/`)**: Human-in-the-loop interface for final validation and guidance.

## Getting Started

### Prerequisites
- Python 3.10+
- Pip

### Installation
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r AGI/requirements.txt
   ```

### Running the System
To run the end-to-end demonstration:
```bash
python -m AGI.src.main
```

### Running Tests
To execute the test suite:
```bash
python -m pytest AGI/tests
```

## Configuration
The swarm behavior (iteration count, agent numbers, pruning thresholds) can be configured within `src/swarm/core.py`. Future updates will move these to a dedicated YAML configuration.

## Roadmap
- [x] Basic Swarm Core & Consensus
- [x] Omnidirectional Memory
- [x] Curiosity-driven Exploration
- [X] HITL Basic Integration
- [ ] Redis-backed scaling for Message Bus
- [ ] Real CLIP-based Visual Cortex integration
