import torch
import torch.nn as nn


class SmallLLM(nn.Module):
    """
    A small, parametrizable, simple LLM architecture for study and visualization.
    
    This class implements a decoder-only transformer architecture with configurable
    components including optional layer normalization, dropout, and softmax output head.
    The model uses learned positional embeddings and a stack of transformer encoder
    layers (used as decoder layers in this context).
    
    Args:
        vocab_size: Size of the vocabulary (default: 1000).
        d_model: Dimension of the model embeddings (default: 128).
        nhead: Number of attention heads (default: 4).
        num_layers: Number of transformer encoder layers (default: 2).
        dim_feedforward: Dimension of the feedforward network (default: 512).
        max_seq_len: Maximum sequence length for positional embeddings (default: 512).
        dropout: Dropout probability. If None, dropout is disabled (default: None).
        layer_norm_eps: Epsilon value for layer normalization (default: 1e-5).
        use_final_layer_norm: Whether to apply layer norm before output projection (default: False).
        use_softmax: Whether to apply softmax to the output logits (default: False).
    
    Example:
        >>> model = SmallLLM(vocab_size=1000, d_model=128, num_layers=2)
        >>> x = torch.randint(0, 1000, (1, 10))  # (batch_size, seq_len)
        >>> output = model(x)  # (batch_size, seq_len, vocab_size)
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        max_seq_len: int = 512,
        dropout: float | None = None,
        layer_norm_eps: float = 1e-5,
        use_final_layer_norm: bool = False,
        use_softmax: bool = False
    ):
        super().__init__()
        
        # store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.use_final_layer_norm = use_final_layer_norm
        self.use_softmax = use_softmax
        
        # token embeddings: map token indices to d_model-dimensional vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # positional embeddings: learnable position encodings
        self.pos_encoding = nn.Embedding(max_seq_len, d_model)
        
        # optional dropout after embedding layer
        self.embedding_dropout = nn.Dropout(dropout) if dropout is not None else nn.Identity()
        
        # transformer encoder layers (used as decoder layers)
        # each layer contains: self-attention, feedforward, layer norm, and dropout
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout if dropout is not None else 0.0,
            layer_norm_eps=layer_norm_eps,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # optional final layer normalization before output projection
        self.final_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps) if use_final_layer_norm else nn.Identity()
        
        # output projection: map hidden states to vocabulary logits
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len) containing token indices.
        
        Returns:
            Output tensor of shape (batch_size, seq_len, vocab_size).
            If use_softmax=True, returns probabilities; otherwise returns logits.
        """
        # get sequence length from input
        seq_len = x.size(1)
        
        # create positional indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        
        # embed tokens and add positional encodings
        x = self.embedding(x) + self.pos_encoding(positions)
        
        # apply optional dropout after embedding
        x = self.embedding_dropout(x)
        
        # pass through transformer layers
        x = self.transformer(x)
        
        # apply optional final layer normalization
        x = self.final_layer_norm(x)
        
        # project to vocabulary size
        logits = self.output(x)
        
        # optionally apply softmax to get probabilities
        if self.use_softmax:
            return torch.softmax(logits, dim=-1)
        
        return logits


model = SmallLLM()
