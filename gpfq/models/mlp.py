import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron (MLP) implementation.

    A simple feedforward neural network with configurable hidden layers,
    activation functions, and dropout.
    """

    def __init__(
        self,
        input_size: int = 28 * 28,
        hidden_sizes: list[int] = [500, 300],
        num_classes: int = 10,
        dropout: float = 0.0,
        bias: bool = True,
        use_batch_norm: bool = False,
    ):
        """Initialize MLP.

        Args:
            input_size: Size of input features
            hidden_sizes: List of hidden layer sizes
            num_classes: Number of output classes
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
        """
        super().__init__()

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_prob = dropout
        self.use_batch_norm = use_batch_norm

        # Build layers
        layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size, bias=bias))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_size = hidden_size

        # Output layer
        layers.append(nn.Linear(in_size, num_classes, bias=bias))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)

        return self.layers(x)
