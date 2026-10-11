"""LoRA adapters merge into full FP32 weights, also for the Laya ModernBERT encoder."""

import torch
from torch import nn

from blockether.vis_decisions._trainer import _cpu_state, _merge_lora, _with_lora


class Layer(nn.Module):
    """The linear layer names of one ModernBERT encoder layer."""

    def __init__(self) -> None:
        super().__init__()
        self.Wqkv = nn.Linear(4, 12)
        self.Wo = nn.Linear(4, 4)
        self.Wi = nn.Linear(4, 8)
        self.norm = nn.LayerNorm(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Wo(self.norm(x)) + self.Wqkv(x)[:, :4] + self.Wi(x)[:, :4]


def test_modernbert_lora_merges_into_plain_layers():
    """#366: Laya LoRA trains only adapters, then leaves plain full weights."""
    torch.manual_seed(0)
    module = Layer()
    names = set(module.state_dict())
    module = _with_lora(module, {"rank": 2, "alpha": 4.0, "dropout": 0.0})
    trainable = {name for name, item in module.named_parameters() if item.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert {name.split(".")[0] for name in trainable} == {"Wqkv", "Wo", "Wi"}
    for name, parameter in module.named_parameters():
        if "lora_B" in name:
            parameter.data.normal_()
    x = torch.randn(3, 4)
    expected = module(x).detach()
    saved = _cpu_state(module)
    assert set(saved) == names
    module = _merge_lora(module)
    assert set(module.state_dict()) == names
    assert all(
        type(module.get_submodule(name)) is nn.Linear for name in ("Wqkv", "Wo", "Wi")
    )
    torch.testing.assert_close(module(x), expected)
    for name, tensor in module.state_dict().items():
        torch.testing.assert_close(tensor, saved[name])
