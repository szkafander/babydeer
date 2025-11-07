import torch
import torch.nn as nn
from typing import Dict, Any

try:
    from torchinfo import summary
except ImportError:
    summary = None


def analyze_computational_complexity(
    model: nn.Module,
    input_shape: tuple[int, int] = (1, 10),
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Analyze computational complexity of a PyTorch model.

    Args:
        model: The PyTorch model to analyze.
        input_shape: Input shape as (batch_size, seq_len) for token-based models.
            For transformer models, this represents (batch_size, sequence_length).
        device: Device to run analysis on ('cpu' or 'cuda').

    Returns:
        Dictionary containing:
            - 'flops_per_token': FLOP count per token (float)
            - 'big_o_complexity': Big O time complexity as string (e.g., "O(n²)")
            - 'total_flops': Total FLOP count for the input shape (int)
            - 'macs': Multiply-accumulate operations (int, if available)
    """
    model.eval()
    batch_size, seq_len = input_shape

    results: Dict[str, Any] = {}

    if summary is not None:
        try:
            model_summary = summary(
                model,
                input_size=input_shape,
                device=device,
                verbose=0,
            )
            macs = model_summary.total_mult_adds
            if macs is not None and macs > 0:
                flops = macs * 2
                total_tokens = batch_size * seq_len
                flops_per_token = flops / total_tokens if total_tokens > 0 else 0.0
                results["macs"] = int(macs)
                results["total_flops"] = int(flops)
                results["flops_per_token"] = flops_per_token
        except Exception:
            pass

    big_o = _determine_big_o_complexity(model, seq_len)
    results["big_o_complexity"] = big_o

    if "flops_per_token" not in results:
        flops_per_token = _estimate_flops_per_token(model, seq_len)
        results["flops_per_token"] = flops_per_token
        results["total_flops"] = flops_per_token * batch_size * seq_len

    return results


def _determine_big_o_complexity(model: nn.Module, seq_len: int) -> str:
    """
    Determine Big O time complexity by analyzing model architecture.

    Args:
        model: The PyTorch model to analyze.
        seq_len: Sequence length used for analysis.

    Returns:
        Big O complexity string (e.g., "O(n²)", "O(n)").
    """
    has_attention = False
    has_transformer = False
    num_layers = 0
    d_model = None

    for name, module in model.named_modules():
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            has_attention = True
            has_transformer = True
            if isinstance(module, nn.TransformerEncoderLayer):
                num_layers += 1
        elif isinstance(module, nn.TransformerEncoder):
            has_transformer = True
            num_layers = len(module.layers)
        elif isinstance(module, nn.TransformerDecoder):
            has_transformer = True
            num_layers = len(module.layers)
        elif isinstance(module, nn.Linear) and d_model is None:
            d_model = module.in_features

    if hasattr(model, "d_model"):
        d_model = model.d_model

    if has_transformer or has_attention:
        if num_layers > 0:
            return f"O(L * n² * d)"
        else:
            return "O(n² * d)"
    else:
        return "O(n * d)"


def _estimate_flops_per_token(model: nn.Module, seq_len: int) -> float:
    """
    Estimate FLOPs per token when torchinfo is not available.

    Args:
        model: The PyTorch model to analyze.
        seq_len: Sequence length.

    Returns:
        Estimated FLOPs per token.
    """
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    has_attention = False
    num_layers = 0
    d_model = 128

    for module in model.modules():
        if isinstance(module, (nn.MultiheadAttention, nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
            has_attention = True
        elif isinstance(module, nn.TransformerEncoder):
            num_layers = len(module.layers)
        elif isinstance(module, nn.TransformerDecoder):
            num_layers = len(module.layers)
        elif isinstance(module, nn.Linear):
            d_model = max(d_model, module.in_features)

    if hasattr(model, "d_model"):
        d_model = model.d_model

    if has_attention:
        attention_flops_per_token = seq_len * d_model * 4
        ffn_flops_per_token = d_model * d_model * 2
        flops_per_token = (attention_flops_per_token + ffn_flops_per_token) * max(num_layers, 1)
    else:
        flops_per_token = total_params * 2

    return float(flops_per_token)

