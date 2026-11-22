"""Model shapes, independent of pretrained checkpoints."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEConfig:
    hidden_size: int = 2048
    intermediate_size: int = 1024
    num_experts: int = 64
    top_k: int = 8
    norm_topk_prob: bool = False

    def __post_init__(self):
        for name in ("hidden_size", "intermediate_size", "num_experts", "top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")

    @classmethod
    def tiny(cls):
        return cls(hidden_size=32, intermediate_size=48, num_experts=8, top_k=2)

    def weight_bytes(self, element_size=2):
        return 3 * self.num_experts * self.hidden_size * self.intermediate_size * element_size
