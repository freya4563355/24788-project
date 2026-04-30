# 24788-project

## Baseline(GCN) and 2 model variant(SchNet and DimeNet) Environment Setup

```bash
pip install torch-geometric
# For SchNet and DimeNet
pip install torch-cluster torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.5.0+cu121.html
```

Note: This project requires a GPU environment. We recommend Google Colab with T4 GPU.
The above torch-cluster installation uses CUDA 12.1; adjust the URL to match your CUDA version.

## Data

QM9 dataset is downloaded automatically by PyTorch Geometric on first run:
```python
from torch_geometric.datasets import QM9
dataset = QM9(root='./data/QM9')
```
No manual download required (~300MB).

## Model Checkpoints

- `best_GCN.pt` — GCN trained for 100 epochs, Val MAE: 300.3 meV
- `best_SchNet.pt` — SchNet trained for 100 epochs, Val MAE: 97.5 meV
- `best_DimeNet.pt` — DimeNet trained for 200 epochs, Val MAE: 80.0 meV

## Reproduce Results

Open `reproduce_result.ipynb` in Google Colab and run all cells.
This will load the saved checkpoints and regenerate metrics without retraining.
