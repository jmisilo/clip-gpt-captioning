from dataclasses import dataclass

# prepare config class with dataclass
@dataclass
class Config:
    seed: int = 100
    num_workers: int = 2
    train_size: int = 0.84
    val_size: int = 0.13
    epochs: int = 300
    lr: int = 6e-3
    k: float = 0.33
    batch_size_exp: int = 6
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    max_len: int = 20
    dropout: float = 0.1
    weights_dir: str = 'weights'