"""
Example script to create and visualize a SmallLLM model.

This script demonstrates how to instantiate a SmallLLM model with various
parameters and generate a visualization of its computation graph.
"""

import sys
from pathlib import Path

# Add parent directory to path to import babydeer
sys.path.insert(0, str(Path(__file__).parent.parent))

from babydeer import SmallLLM, render_graph


def main():
    """Create a SmallLLM model and visualize its computation graph."""
    # Define model parameters for demonstration
    vocab_size = 1000
    d_model = 128
    nhead = 4
    num_layers = 2
    dim_feedforward = 512
    dropout = 0.1
    use_final_layer_norm = True
    use_softmax = False
    
    # Create the model
    print("Creating SmallLLM model...")
    model = SmallLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_final_layer_norm=use_final_layer_norm,
        use_softmax=use_softmax
    )
    
    # Create filename with parameter names and values
    # Replace dots in dropout value with underscore for cleaner filename
    dropout_str = str(dropout).replace('.', '_') if dropout is not None else 'None'
    filename_parts = [
        f"vocab{vocab_size}",
        f"dmodel{d_model}",
        f"nhead{nhead}",
        f"layers{num_layers}",
        f"ffn{dim_feedforward}",
        f"dropout{dropout_str}",
        f"layernorm{use_final_layer_norm}",
        f"softmax{use_softmax}"
    ]
    filename = f"torch_decoder_only_{'_'.join(filename_parts)}"
    
    # Set output path
    output_dir = Path(__file__).parent / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    
    print(f"Generating visualization...")
    print(f"Output will be saved to: {output_path}.svg")
    
    # Render the graph
    render_graph(
        model=model,
        depth=6,
        expand_nested=True,
        hide_inner_tensors=False,
        hide_module_functions=False,
        batch_size=1,
        seq_len=10,
        device='cpu',
        filename=str(output_path),
        format='svg'
    )
    
    print(f"✓ Visualization saved to {output_path}.svg")


if __name__ == "__main__":
    main()

