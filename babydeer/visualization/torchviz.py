import torch
import torch.nn as nn

from torchview import draw_graph


def render_graph(
    model: nn.Module,
    depth: int = 6,
    expand_nested: bool = True,
    hide_inner_tensors: bool = False,
    hide_module_functions: bool = False,
    input_data: torch.Tensor | None = None,
    input_size: tuple | None = None,
    batch_size: int = 1,
    seq_len: int = 10,
    device: str = 'cpu',
    filename: str = "model_detailed",
    format: str = "svg",
    **kwargs
) -> None:
    """
    Render a PyTorch model's computation graph using torchview.
    
    Args:
        model: The PyTorch model to visualize.
        depth: The depth of nested modules to display.
        expand_nested: Whether to expand nested modules.
        hide_inner_tensors: Whether to hide inner tensors in the graph.
        hide_module_functions: Whether to hide module functions in the graph.
        input_data: Direct input tensor(s) for the model. If None, will be inferred.
        input_size: Shape of input tensor(s) as a tuple. If None, will be inferred.
        batch_size: Batch size for input data (used when inferring input shape).
        seq_len: Sequence length for input data (used when inferring input shape).
        device: Device to run the model on ('cpu' or 'cuda').
        filename: Output filename (without extension).
        format: Output format ('svg', 'png', 'pdf', etc.).
        **kwargs: Additional keyword arguments passed to draw_graph.
    """
    # determine input_data or input_size
    if input_data is None and input_size is None:
        # try to infer input shape from the model
        vocab_size = None
        
        # check for embedding layer to get vocab_size
        for module in model.modules():
            if isinstance(module, nn.Embedding):
                vocab_size = module.num_embeddings
                break
        
        # if no embedding found, check for common patterns
        if vocab_size is None:
            # check if model has vocab_size attribute
            if hasattr(model, 'vocab_size'):
                vocab_size = model.vocab_size
            else:
                # Default fallback
                vocab_size = 1000
        
        # create input_data based on model's expected input shape
        # for decoder-only models, input is typically (batch_size, seq_len)
        input_data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    # create the graph
    model_graph = draw_graph(
        model,
        input_data=input_data,
        input_size=input_size,
        device=device,
        depth=depth,
        expand_nested=expand_nested,
        hide_inner_tensors=hide_inner_tensors,
        hide_module_functions=hide_module_functions,
        **kwargs
    )
    
    # render the graph
    model_graph.visual_graph.render(filename, format=format)
