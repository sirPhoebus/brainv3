# Omnidirectional Swarm System (AGI)

A modular AI architecture designed for reasoning through an omnidirectional swarm of agents. This system features a visual processing layer, an embedding language middleware, and a decentralized swarm core with curiosity-driven exploration and human-in-the-loop verification.

## Architecture

- **Visual Cortex (`src/cortex/`)**: Uses OpenAI's **CLIP** model to segment images into patches and generate latent embeddings.
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
- PyTorch & Transformers (Auto-installed via requirements)

### Installation
1. Clone the repository.
2. Create and activate the dedicated virtual environment:
   ```powershell
   cd AGI
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   cd ..
   ```

### Running the System
To run the end-to-end demonstration with real CLIP processing (from the project root):
```powershell
.\AGI\.venv\Scripts\python -m AGI.src.main
```

### Running Tests
To execute the test suite (from the project root):
```powershell
.\AGI\.venv\Scripts\python -m pytest AGI/tests
```

## How it Works (Phase 1)
The current implementation replaces all mock systems with actual ML components:
1. `CLIPVisualCortex` loads a Vision Transformer and processes `AGI/examples/sample_image.png`.
2. It generates 49 patches (7x7 grid) and embeds them.
3. Swarm agents ingest these patches, propose detection hypotheses, and reach consensus after multiple reasoning iterations.

## Roadmap
- [x] Basic Swarm Core & Consensus
- [x] Omnidirectional Memory
- [x] Curiosity-driven Exploration
- [x] HITL Basic Integration
- [x] **Real CLIP-based Visual Cortex integration**
- [ ] Redis-backed scaling for Message Bus
- [ ] Multi-turn interactive HITL feedback loop
- [ ] Logic puzzle solving engine
