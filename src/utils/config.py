from dataclasses import dataclass

# prepare config class with dataclass

@dataclass
class Config:
    seed: int = 100
    num_workers: int = 4
    train_size: int = 0.8
    val_size: int = 0.1
    epochs: int = 50
    lr: int = 2e-5
    k: float = 0.3
    batch_size_exp: int = 1
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 20
    dropout: float = 0.1