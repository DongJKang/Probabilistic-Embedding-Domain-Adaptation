# Probabilistic Embedding Domain Adaptation
This repository provides a PyTorch implementation of a probabilistic embedding-based domain adaptation framework. The approach aims to enhance model generalization across different domains by leveraging probabilistic embeddings and unsupervised domain adaptation.

# Features
- **Probabilistic Embeddings**: 
Utilizes probabilistic feature representations (e.g., Gaussian, Laplace, Cauchy) to model uncertainty and improve generalization under domain shift.

- **⚡ PyTorch Lightning Integration**: 
Built on top of PyTorch Lightning for modular, scalable, and hardware-agnostic training workflows.

- **Unsupervised Domain Adaptation (UDA)**: 
Supports six UDA strategies:
  - CORAL (Correlation Alignment)
  - MMD (Maximum Mean Discrepancy)
  - DANN (Domain-Adversarial Neural Network) 
  - ADDA (Adversarial Discriminative Domain Adaptation)  
  - MCD (Maximum Classifier Discrepancy)  
  - HHD (Hypothesis-based Hybrid Discrepancy)  

# Installation

1. **Clone the repository**:

```bash
git clone https://github.com/DongJKang/Probabilistic-Embedding-Domain-Adaptation.git
cd Probabilistic-Embedding-Domain-Adaptation

```

2. **Create a virtual environment (optional but recommended)**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

```

3. **Install depenedencies**:
```bash
pip install -r requirements.txt

```

# Usage

1. **Configure the experiment**:
- Modify the configuration files in the `configs/` directory to set up your experiment parameters.

2. **Run training**:
```bash
python main.py --config configs/your_config.yaml

```

3. **Monitor training**:
- Training logs and checkpoints will be saved in the `logs/` directory.

# Project Structure

- `configs/`: Configuration files for different experiments.
- `lightning_modules/`: PyTorch Lightning modules defining the training logic.
- `models/`: Model architectures used in the experiments.
- `utils/`: Utility functions.
- `main.py`: Entry point for training and evaluation.
- `requirements.txt`: List of Python dependencies.
- `pyproject.toml`: Project metadata and build configuration.