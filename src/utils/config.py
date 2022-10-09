from dataclasses import dataclass

# prepare config class with dataclass

@dataclass
class Config:
    num_layers: int = 6
    n_heads: int = 16
    forward_expansion: int = 4
    dropout: float = 0.1
    max_len: int = 20
    batch_size_exp: int = 1
    epochs: int = 50
    lr: int = 2e-5
    k: float = 0.3