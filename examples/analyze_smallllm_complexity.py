"""Example script to analyze computational complexity of SmallLLM."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from babydeer import SmallLLM
from babydeer.evaluation import analyze_computational_complexity


def format_number(num: float | int) -> str:
    """Format large numbers with appropriate units."""
    if num >= 1e12:
        return f"{num / 1e12:.2f}T"
    elif num >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif num >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f}K"
    else:
        return f"{num:.2f}"


def print_complexity_results(results: dict, model_name: str) -> None:
    """Print formatted complexity analysis results."""
    print(f"\n{'=' * 60}")
    print(f"Computational Complexity Analysis: {model_name}")
    print(f"{'=' * 60}\n")

    if "big_o_complexity" in results:
        print(f"Big O Time Complexity: {results['big_o_complexity']}")
        print()

    if "flops_per_token" in results:
        flops_per_token = results["flops_per_token"]
        print(f"FLOPs per Token: {format_number(flops_per_token)} ({flops_per_token:,.0f})")
        print()

    if "total_flops" in results:
        total_flops = results["total_flops"]
        print(f"Total FLOPs: {format_number(total_flops)} ({total_flops:,.0f})")
        print()

    if "macs" in results:
        macs = results["macs"]
        print(f"MACs (Multiply-Accumulate): {format_number(macs)} ({macs:,.0f})")
        print()

    print(f"{'=' * 60}\n")


def main() -> None:
    """Analyze computational complexity of SmallLLM and save results."""
    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    dropout = 0.1
    use_final_layer_norm = True
    use_softmax = False

    model = SmallLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_final_layer_norm=use_final_layer_norm,
        use_softmax=use_softmax,
    )

    input_shape = (1, 10)
    batch_size, seq_len = input_shape

    print(f"Analyzing model with input shape: {input_shape}")
    print(f"Model parameters: vocab={vocab_size}, d_model={d_model}, layers={num_layers}")

    results = analyze_computational_complexity(model, input_shape=input_shape, device="cpu")

    model_config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "nhead": nhead,
        "num_layers": num_layers,
        "dim_feedforward": dim_feedforward,
        "dropout": dropout,
        "use_final_layer_norm": use_final_layer_norm,
        "use_softmax": use_softmax,
    }

    output_data = {
        "model": "SmallLLM",
        "model_config": model_config,
        "input_shape": {"batch_size": batch_size, "seq_len": seq_len},
        "complexity_analysis": results,
    }

    print_complexity_results(results, "SmallLLM")

    output_dir = Path(__file__).parent / "assessments"
    output_dir.mkdir(parents=True, exist_ok=True)

    filename_parts = [
        f"vocab{vocab_size}",
        f"dmodel{d_model}",
        f"nhead{nhead}",
        f"layers{num_layers}",
        f"ffn{dim_feedforward}",
        f"dropout{str(dropout).replace('.', '_')}",
        f"layernorm{use_final_layer_norm}",
        f"softmax{use_softmax}",
    ]
    filename = f"complexity_{'_'.join(filename_parts)}.json"
    output_path = output_dir / filename

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()

