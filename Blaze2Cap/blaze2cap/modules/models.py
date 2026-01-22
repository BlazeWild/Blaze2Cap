import torch
import torch.nn as nn
import math

class LayerNorm(nn.LayerNorm):
    """
    Subclass torch's LayerNorm to handle fp16.
    Forces the normalization to run in float32 to prevent overflow/underflow,
    then casts back to the original dtype (fp16/bf16).
    """
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        # Always upcast to float32 for the normalization stats
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    
    
class QuickGELU(nn.Module):
    """
    Quick Gaussian Error Linear Unit activation function.
    """
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)
    
class FeedForward(nn.Module):
    """
    Standard Point-wise Feed Forward Network (FFN).
    Input -> Linear -> GELU -> Dropout -> Linear -> Output
    """
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.activation = QuickGELU()
        self.dropout = nn.Dropout(dropout)
        self.w_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class CausalSelfAttention(nn.Module):
    """
    Calculates Attention with a fixed Triangular Mask (Causal) 
    AND a variable Padding Mask (Virtual Wall).
    """
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_head,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        B, N, C = x.shape
        
        # generate causal mask (time constraints)
        # prevents looking at future tokens/frames
        causal_mask = torch.triu(torch.ones(N,N)* float('-inf', diagonal=1).to(x.device))
        
        out, weights = self.attn(
            query =x, 
            key = x,
            value = x,
            attn_mask = causal_mask,
            key_padding_mask = key_padding_mask,
            need_weights = True
        )
        
        return self.dropout(out), weights
    
class TransformerBlock(nn.Module):
    """
    A single layer containing the full Residual connection logic.
    Structure: Pre-Norm -> Attention -> Add -> Pre-Norm -> MLP -> Add
    """
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # attention
        self.norm1 = LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        # feed forward
        self.norm2 = LayerNorm(d_model)
        self.mlp = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, key_padding_mask=None):
        # residual + attention
        residual = x #  residual
        x_norm = self.norm1(x) #pre layer norm
        # attnention out and weights
        attn_out, weights = self.attn(x_norm, key_padding_mask=key_padding_mask) # calulate attention
        x = residual + attn_out #residual connection
 
        # another residual + mlp
        residual = x
        x_norm = self.norm2(x) # pre layer norm
        mlp_out = self.mlp(x_norm) # feed forward
        x = residual + self.dropout(mlp_out) # residual connection
        
        #return output and attention weights
        return x, weights
    
class TemporalTransfomerEncoder(nn.Module):
    """ Stacking n_layers of TransformerBlock modules. """
    def __init__(self, num_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout) for _ in range(num_layers)
        ])
        # final prenorm 
        self.norm_final = LayerNorm(d_model)
        
    def forward(self, x, key_padding_mask=None, return_all_weights = False):
        all_weights = []
        for layer in self.layers:
            x, weights = layer(x, key_padding_mask=key_padding_mask)
            if return_all_weights:
                all_weights.append(weights)
        
        x = self.norm_final(x)
        
        if return_all_weights:
            return x, all_weights
        return x