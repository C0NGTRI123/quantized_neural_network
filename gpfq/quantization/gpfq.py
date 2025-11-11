# --------------------------------------------------------------
#  QuantizedNeuralNetwork + QuantizedCNN  (pure-PyTorch, .pt files)
# --------------------------------------------------------------
import concurrent.futures
import copy
import itertools
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ------------------------------------------------------------------
#  Alphabet helper
# ------------------------------------------------------------------
def make_alphabet(bits: float, scalar: float, layer_weights: torch.Tensor) -> np.ndarray:
    n = int(round(2**bits))
    base = np.linspace(-1.0, 1.0, n)

    # Use a better scaling approach - consider both median and max for better range coverage
    weights_np = layer_weights.detach().cpu().numpy().flatten()
    weights_abs = np.abs(weights_np)

    # Use a combination of median and percentile for better scaling
    median_val = np.median(weights_abs)
    rad = scalar * median_val
    return rad * base


# ------------------------------------------------------------------
#  JIT-compiled greedy step (same math as the paper)
# ------------------------------------------------------------------
@torch.jit.script
def _greedy_step(
    w: torch.Tensor,  # scalar weight for this entry
    u: torch.Tensor,  # (M,)  current error
    X: torch.Tensor,  # (M,)  analog direction
    Xq: torch.Tensor,  # (M,)  quantized direction
    alphabet: torch.Tensor,  # (K,)
) -> tuple[torch.Tensor, torch.Tensor]:
    norm_Xq = Xq.norm(p=2)
    if norm_Xq < 1e-16:
        q = torch.tensor(0.0, device=w.device)
    else:
        proj = torch.dot(Xq, u + w * X) / (norm_Xq**2)
        if torch.abs(torch.dot(Xq, u)) < 1e-10:
            q = alphabet[torch.argmin(torch.abs(alphabet - w))]
        else:
            q = alphabet[torch.argmin(torch.abs(alphabet - proj))]
    u = u + w * X - q * Xq
    return q, u


# ------------------------------------------------------------------
#  Workers – they receive NumPy arrays (CPU) and return NumPy
# ------------------------------------------------------------------
def _quantize_layer_weights(args):
    W_flat, pt_path, alphabet = args  # Flattened weight matrix
    device = torch.device("cpu")
    W_flat = torch.from_numpy(W_flat).to(device)
    alphabet = torch.from_numpy(alphabet).to(device)

    data = torch.load(pt_path)
    wX = data["wX"].to(device)  # shape: (batch_size, input_features)
    qX = data["qX"].to(device)  # shape: (batch_size, input_features)

    batch_size, input_features = wX.shape
    num_weights = len(W_flat)

    # Initialize error and quantized weights
    u = torch.zeros(batch_size, device=device)
    Q_flat = torch.zeros_like(W_flat)

    # GPFQ algorithm: process weights sequentially
    for t in range(num_weights):
        # Get the corresponding input feature index for this weight
        input_idx = t % input_features

        # Get activations for this input feature
        X = wX[:, input_idx]  # shape: (batch_size,)
        Xq = qX[:, input_idx]  # shape: (batch_size,)

        # Quantize this weight using the greedy step
        Q_flat[t], u = _greedy_step(W_flat[t], u, X, Xq, alphabet)

    return Q_flat.cpu().numpy()


def _quantize_conv_filter(args):
    filt_fp, channel_idx, pt_path, alphabet = args
    device = torch.device("cpu")
    filt_fp = torch.from_numpy(filt_fp).to(device)
    alphabet = torch.from_numpy(alphabet).to(device)

    data = torch.load(pt_path)  # dict: {f"wX_c{channel_idx}":..., f"qX_c{channel_idx}":...}
    key_w = f"wX_c{channel_idx}"
    key_q = f"qX_c{channel_idx}"
    N = data[key_w].shape[0]  # number of patches for this channel
    patch_dim = data[key_w].shape[1]
    u = torch.zeros(patch_dim, device=device)
    q = torch.zeros_like(filt_fp)

    idx = 0
    for ky in range(filt_fp.shape[0]):
        for kx in range(filt_fp.shape[1]):
            if idx >= N:
                break
            X = data[key_w][idx].to(device)
            Xq = data[key_q][idx].to(device)
            q[ky, kx], u = _greedy_step(filt_fp[ky, kx], u, X, Xq, alphabet)
            idx += 1
    return q.cpu().numpy()


# ------------------------------------------------------------------
#  Base class – works for any nn.Module
# ------------------------------------------------------------------
class QuantizedNeuralNetwork(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        mini_batch_size: int = 32,
        logger: logging.Logger = None,
        ignore_layers: list[int] = None,
        bits: float = np.log2(3),
        alphabet_scalar: float = 1.0,
    ):
        super().__init__()
        self.model = model
        self.dataloader = dataloader
        self.mini_batch_size = mini_batch_size
        self.logger = logger or logging.getLogger(__name__)
        self.ignore_layers = ignore_layers or []
        self.bits = bits
        self.alphabet_scalar = alphabet_scalar
        self.device = next(model.parameters()).device

        # Create a deep copy instead of JIT tracing to avoid static graph issues
        self.quantized = copy.deepcopy(model)
        self.quantized.to(self.device)

    # ------------------------------------------------------------------
    #  Logging helper
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    # ------------------------------------------------------------------
    #  Run both networks up to *layer_idx* and return activations
    # ------------------------------------------------------------------
    def _forward_up_to(self, x: torch.Tensor, upto: int, model: nn.Module = None) -> torch.Tensor:
        if model is None:
            model = self.model

        x = x.to(self.device)

        # Handle different model architectures
        if hasattr(model, "layers") and isinstance(model.layers, nn.Sequential):
            # For models with a 'layers' attribute (like MLP)
            for i, layer in enumerate(model.layers):
                if i > upto:
                    break
                x = layer(x)
        else:
            # For other models, iterate through children
            for i, module in enumerate(model.children()):
                if i > upto:
                    break
                x = module(x)
        return x

    # ------------------------------------------------------------------
    #  Dump activations for a layer into a .pt file
    # ------------------------------------------------------------------
    def _dump_layer_activations(self, layer_idx: int, pt_path: str):
        os.makedirs(os.path.dirname(pt_path), exist_ok=True)
        wX_list, qX_list = [], []

        self.model.eval()
        self.quantized.eval()

        with torch.no_grad():
            for batch in self.dataloader:
                x, _ = batch
                x = x.to(self.device)

                # Flatten input for MLP models
                if hasattr(self.model, "layers") and x.dim() > 2:
                    x = x.view(x.size(0), -1)

                wX = self._forward_up_to(x, layer_idx - 1, self.model) if layer_idx > 0 else x
                qX = self._forward_up_to(x, layer_idx - 1, self.quantized) if layer_idx > 0 else x
                wX_list.append(wX.detach().cpu())
                qX_list.append(qX.detach().cpu())

        torch.save({"wX": torch.cat(wX_list, dim=0), "qX": torch.cat(qX_list, dim=0)}, pt_path)

    # ------------------------------------------------------------------
    #  Quantize a Linear layer
    # ------------------------------------------------------------------
    def _quantize_linear(self, layer_idx: int, layer: nn.Linear):
        W = layer.weight.detach().cpu().numpy()  # (out, in)
        pt_path = f"./temp/layer{layer_idx}_act.pt"
        self._dump_layer_activations(layer_idx, pt_path)

        alphabet = make_alphabet(self.bits, self.alphabet_scalar, layer.weight)
        Q = np.zeros_like(W)

        self._log(f"\tQuantizing layer with {W.size} weights (Linear) …")
        tic = time.time()

        # Flatten the weight matrix for sequential processing
        W_flat = W.flatten()
        Q_flat = _quantize_layer_weights((W_flat, pt_path, alphabet))
        Q = Q_flat.reshape(W.shape)

        # write back to the quantized model
        with torch.no_grad():
            if hasattr(self.quantized, "layers") and isinstance(self.quantized.layers, nn.Sequential):
                # For models with 'layers' attribute
                target_layer = self.quantized.layers[layer_idx]
                if isinstance(target_layer, nn.Linear):
                    target_layer.weight.copy_(torch.from_numpy(Q).to(self.device))
            else:
                # For other model structures
                layers = list(self.quantized.modules())
                if layer_idx < len(layers) and isinstance(layers[layer_idx], nn.Linear):
                    layers[layer_idx].weight.copy_(torch.from_numpy(Q).to(self.device))

        if os.path.exists(pt_path):
            os.remove(pt_path)
        self._log(f"\tdone in {time.time() - tic:.2f}s")

    # ------------------------------------------------------------------
    #  Public entry point
    # ------------------------------------------------------------------
    def quantize_network(self):
        # Get the correct layer structure based on model type
        if hasattr(self.model, "layers") and isinstance(self.model.layers, nn.Sequential):
            # For models with a 'layers' attribute (like MLP)
            layers = list(self.model.layers)
            for idx, m in enumerate(layers):
                if isinstance(m, nn.Linear) and idx not in self.ignore_layers:
                    self._log(f"Quantizing Linear layer {idx} …")
                    self._quantize_linear(idx, m)
        else:
            # For other models, use the original approach
            layers = list(self.model.modules())
            for idx, m in enumerate(layers):
                if isinstance(m, nn.Linear) and idx not in self.ignore_layers:
                    self._log(f"Quantizing Linear layer {idx} …")
                    self._quantize_linear(idx, m)


# ------------------------------------------------------------------
#  CNN extension – Conv2d + patch extraction
# ------------------------------------------------------------------
class QuantizedCNN(QuantizedNeuralNetwork):
    def __init__(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        mini_batch_size: int = 32,
        patch_mini_batch_size: int = 5000,
        logger: logging.Logger = None,
        bits: float = np.log2(3),
        alphabet_scalar: float = 1.0,
        quantize_conv: bool = True,
    ):
        super().__init__(model, dataloader, mini_batch_size, logger, [], bits, alphabet_scalar)
        self.patch_batch = patch_mini_batch_size
        self.quantize_conv = quantize_conv

    # ------------------------------------------------------------------
    #  Unfold a single channel into patches
    # ------------------------------------------------------------------
    @staticmethod
    def _unfold_channel(x: torch.Tensor, ks, stride, pad, dil):
        """X : (B, 1, H, W)
        returns (B * out_h * out_w, C*kh*kw)  –  C==1 here
        """
        return F.unfold(x, kernel_size=ks, stride=stride, padding=pad, dilation=dil).transpose(1, 2)

    # ------------------------------------------------------------------
    #  Build per-channel patch tensors → .pt
    # ------------------------------------------------------------------
    def _dump_conv_patches(self, layer_idx: int, conv: nn.Conv2d, act_pt: str) -> str:
        ks = conv.kernel_size
        stride = conv.stride
        pad = conv.padding
        dil = conv.dilation
        in_ch = conv.in_channels

        device = next(self.model.parameters()).device

        patch_pt = f"./temp/layer{layer_idx}_patches.pt"
        patch_dict = {}

        processed = 0
        total = len(self.dataloader.dataset)
        while processed < total:
            cur_bs = min(self.patch_batch, total - processed)
            batch = next(itertools.islice(self.dataloader, cur_bs), None)
            if batch is None:
                break
            x, _ = batch
            x = x.to(device)
            wX = self._forward_up_to(x, layer_idx - 1, self.model) if layer_idx > 0 else x
            qX = self._forward_up_to(x, layer_idx - 1, self.quantized) if layer_idx > 0 else x

            for c in range(in_ch):
                w_c = wX[:, c : c + 1, :, :].cpu()
                q_c = qX[:, c : c + 1, :, :].cpu()
                pw = self._unfold_channel(w_c, ks, stride, pad, dil)
                pq = self._unfold_channel(q_c, ks, stride, pad, dil)

                key_w = f"wX_c{c}"
                key_q = f"qX_c{c}"
                if key_w not in patch_dict:
                    patch_dict[key_w] = [pw]
                    patch_dict[key_q] = [pq]
                else:
                    patch_dict[key_w].append(pw)
                    patch_dict[key_q].append(pq)

            processed += x.shape[0]

        # concatenate per-channel
        for c in range(in_ch):
            key_w = f"wX_c{c}"
            key_q = f"qX_c{c}"
            patch_dict[key_w] = torch.cat(patch_dict[key_w], dim=0)
            patch_dict[key_q] = torch.cat(patch_dict[key_q], dim=0)

        torch.save(patch_dict, patch_pt)
        return patch_pt

    # ------------------------------------------------------------------
    #  Quantize a Conv2d layer
    # ------------------------------------------------------------------
    def _quantize_conv(self, layer_idx: int, conv: nn.Conv2d):
        W = conv.weight.detach().cpu().numpy()  # (out, in, kh, kw)
        act_pt = f"./temp/layer{layer_idx}_act.pt"
        self._dump_layer_activations(layer_idx, act_pt)

        patch_pt = self._dump_conv_patches(layer_idx, conv, act_pt)
        alphabet = make_alphabet(self.bits, self.alphabet_scalar, conv.weight)
        Q = np.zeros_like(W)

        self._log(f"\tQuantizing {W.shape[0]} filters × {W.shape[1]} channels …")
        tic = time.time()
        for c in range(W.shape[1]):  # per input channel
            with concurrent.futures.ProcessPoolExecutor() as pool:
                tasks = [(W[f, c], c, patch_pt, alphabet) for f in range(W.shape[0])]
                for f, q_filt in enumerate(pool.map(_quantize_conv_filter, tasks)):
                    Q[f, c] = q_filt

        # write back to the quantized model
        with torch.no_grad():
            if hasattr(self.quantized, "layers") and isinstance(self.quantized.layers, nn.Sequential):
                # For models with 'layers' attribute
                target_layer = self.quantized.layers[layer_idx]
                if isinstance(target_layer, nn.Conv2d):
                    target_layer.weight.copy_(torch.from_numpy(Q).to(self.device))
            else:
                # For other model structures
                layers = list(self.quantized.modules())
                if layer_idx < len(layers) and isinstance(layers[layer_idx], nn.Conv2d):
                    layers[layer_idx].weight.copy_(torch.from_numpy(Q).to(self.device))

        for p in [act_pt, patch_pt]:
            if os.path.exists(p):
                os.remove(p)
        self._log(f"\tdone in {time.time() - tic:.2f}s")

    # ------------------------------------------------------------------
    #  Override – handle both Linear and Conv2d
    # ------------------------------------------------------------------
    def quantize_network(self):
        # Get the correct layer structure based on model type
        if hasattr(self.model, "layers") and isinstance(self.model.layers, nn.Sequential):
            # For models with a 'layers' attribute (like MLP in CNN form)
            layers = list(self.model.layers)
            for idx, m in enumerate(layers):
                if isinstance(m, nn.Linear) and idx not in self.ignore_layers:
                    self._log(f"Quantizing Linear layer {idx} …")
                    self._quantize_linear(idx, m)
                elif isinstance(m, nn.Conv2d) and self.quantize_conv:
                    self._log(f"Quantizing Conv2d layer {idx} …")
                    self._quantize_conv(idx, m)
        else:
            # For other models, use the original approach
            layers = list(self.model.modules())
            for idx, m in enumerate(layers):
                if isinstance(m, nn.Linear) and idx not in self.ignore_layers:
                    self._log(f"Quantizing Linear layer {idx} …")
                    self._quantize_linear(idx, m)
                elif isinstance(m, nn.Conv2d) and self.quantize_conv:
                    self._log(f"Quantizing Conv2d layer {idx} …")
                    self._quantize_conv(idx, m)
