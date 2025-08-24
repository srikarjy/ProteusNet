"""
Physics-informed attention mechanism that incorporates protein physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PhysicsInformedAttention(nn.Module):
    """
    Novel attention mechanism that incorporates protein physics constraints.
    
    Key innovations:
    1. Distance-aware attention bias
    2. Amino acid interaction potentials
    3. Secondary structure awareness
    4. Learnable physics constants
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize physics-informed attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.dropout = dropout
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Physics-aware components
        self.distance_bias = nn.Parameter(torch.randn(1, n_heads, 1, 1) * 0.02)
        self.interaction_matrix = nn.Parameter(torch.randn(25, 25) * 0.02)  # 25 for vocab size
        self.secondary_structure_bias = nn.Linear(8, n_heads, bias=False)  # 8 SS types
        
        # Learnable physics constants
        self.physics_scale = nn.Parameter(torch.tensor(1.0))
        self.distance_decay = nn.Parameter(torch.tensor(2.0))
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters using Xavier/Glorot initialization."""
        # Initialize projection layers
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
        # Initialize secondary structure bias
        nn.init.xavier_uniform_(self.secondary_structure_bias.weight)
        
        # Initialize interaction matrix symmetrically
        with torch.no_grad():
            self.interaction_matrix.data = (self.interaction_matrix.data + 
                                          self.interaction_matrix.data.T) / 2
    
    def forward(self, 
                x: torch.Tensor,
                physics_features: Optional[torch.Tensor] = None,
                amino_acid_ids: Optional[torch.Tensor] = None,
                secondary_structure: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass of physics-informed attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            physics_features: Optional physics features [batch_size, physics_dim]
            amino_acid_ids: Optional amino acid token IDs [batch_size, seq_len]
            secondary_structure: Optional secondary structure one-hot [batch_size, seq_len, 8]
            mask: Optional attention mask [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            Optionally attention weights if return_attention=True
        """
        batch_size, seq_len, d_model = x.shape
        
        # Standard attention projections
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add physics-informed bias
        if physics_features is not None:
            physics_bias = self._compute_physics_bias(
                physics_features, seq_len, batch_size, amino_acid_ids, secondary_structure
            )
            attention_scores = attention_scores + physics_bias
        
        # Apply attention mask
        if mask is not None:
            attention_scores = self._apply_mask(attention_scores, mask)
        
        # Softmax and dropout
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, v)
        
        # Reshape and project output
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.out_proj(context)
        output = self.output_dropout(output)
        
        if return_attention:
            return output, attention_probs
        return output
    
    def _compute_physics_bias(self, 
                            physics_features: torch.Tensor, 
                            seq_len: int, 
                            batch_size: int,
                            amino_acid_ids: Optional[torch.Tensor] = None,
                            secondary_structure: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute physics-informed attention bias based on protein properties.
        
        Args:
            physics_features: Physics features tensor [batch_size, feature_dim]
            seq_len: Sequence length
            batch_size: Batch size
            amino_acid_ids: Optional amino acid token IDs [batch_size, seq_len]
            secondary_structure: Optional secondary structure labels [batch_size, seq_len, 8]
            
        Returns:
            Physics bias tensor [batch_size, n_heads, seq_len, seq_len]
        """
        device = physics_features.device
        
        # Initialize bias tensor
        physics_bias = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=device)
        
        # 1. Distance-based bias (if physics_features contains distance info)
        distance_bias = self._compute_distance_bias(physics_features, seq_len, batch_size, device)
        physics_bias = physics_bias + distance_bias
        
        # 2. Amino acid interaction bias
        if amino_acid_ids is not None:
            interaction_bias = self._compute_interaction_bias(amino_acid_ids, seq_len, batch_size, device)
            physics_bias = physics_bias + interaction_bias
        
        # 3. Secondary structure bias
        if secondary_structure is not None:
            structure_bias = self._compute_structure_bias(secondary_structure, seq_len, batch_size, device)
            physics_bias = physics_bias + structure_bias
        
        # 4. Add learnable distance bias parameter
        physics_bias = physics_bias + self.distance_bias
        
        return physics_bias
    
    def _compute_distance_bias(self, 
                             physics_features: torch.Tensor, 
                             seq_len: int, 
                             batch_size: int, 
                             device: torch.device) -> torch.Tensor:
        """
        Compute distance-based attention bias.
        
        Args:
            physics_features: Physics features tensor
            seq_len: Sequence length
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Distance bias tensor [batch_size, n_heads, seq_len, seq_len]
        """
        distance_bias = torch.zeros(batch_size, self.n_heads, seq_len, seq_len, device=device)
        
        # Check if physics_features contains distance matrix
        if physics_features.shape[-1] >= seq_len * seq_len:
            # Extract distance matrix from physics features
            distance_matrix = physics_features[:, :seq_len*seq_len].view(batch_size, seq_len, seq_len)
            
            # Apply distance decay with learnable parameters
            # Clamp distance to avoid numerical issues
            clamped_distance = torch.clamp(distance_matrix, min=0.1, max=50.0)
            
            # Apply learnable distance decay function
            decay_power = torch.clamp(self.distance_decay, min=0.1, max=5.0)
            distance_penalty = -torch.pow(clamped_distance, decay_power)
            
            # Scale by learnable physics scale parameter
            scaled_penalty = distance_penalty * torch.sigmoid(self.physics_scale)
            
            # Broadcast across attention heads
            distance_bias = scaled_penalty.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        return distance_bias
    
    def _compute_interaction_bias(self, 
                                amino_acid_ids: torch.Tensor, 
                                seq_len: int, 
                                batch_size: int, 
                                device: torch.device) -> torch.Tensor:
        """
        Compute amino acid interaction bias using learnable interaction matrix.
        
        Args:
            amino_acid_ids: Amino acid token IDs [batch_size, seq_len]
            seq_len: Sequence length
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Interaction bias tensor [batch_size, n_heads, seq_len, seq_len]
        """
        # Ensure interaction matrix is symmetric
        symmetric_matrix = self.get_interaction_matrix()
        
        # Clamp amino acid IDs to valid range
        clamped_ids = torch.clamp(amino_acid_ids, 0, symmetric_matrix.shape[0] - 1)
        
        # Create pairwise interaction matrix for each sequence
        # [batch_size, seq_len, seq_len]
        interaction_scores = torch.zeros(batch_size, seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Get interaction scores for all pairs (i,j) across batch
                aa_i = clamped_ids[:, i]  # [batch_size]
                aa_j = clamped_ids[:, j]  # [batch_size]
                
                # Look up interaction scores from symmetric matrix
                interaction_scores[:, i, j] = symmetric_matrix[aa_i, aa_j]
        
        # Apply distance-dependent scaling (closer residues have stronger interactions)
        position_matrix = torch.arange(seq_len, device=device).float()
        position_diff = torch.abs(position_matrix.unsqueeze(0) - position_matrix.unsqueeze(1))
        
        # Distance decay for interactions (1/(1+d) scaling)
        distance_scaling = 1.0 / (1.0 + position_diff)
        distance_scaling = distance_scaling.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Apply distance scaling to interaction scores
        scaled_interactions = interaction_scores * distance_scaling
        
        # Broadcast across attention heads
        interaction_bias = scaled_interactions.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        return interaction_bias
    
    def _compute_structure_bias(self, 
                              secondary_structure: torch.Tensor, 
                              seq_len: int, 
                              batch_size: int, 
                              device: torch.device) -> torch.Tensor:
        """
        Compute secondary structure-based attention bias.
        
        Args:
            secondary_structure: Secondary structure one-hot [batch_size, seq_len, 8]
            seq_len: Sequence length
            batch_size: Batch size
            device: Device for tensor operations
            
        Returns:
            Structure bias tensor [batch_size, n_heads, seq_len, seq_len]
        """
        # Project secondary structure to attention heads
        # [batch_size, seq_len, n_heads]
        structure_proj = self.secondary_structure_bias(secondary_structure)
        
        # Create pairwise structure compatibility matrix
        # [batch_size, seq_len, seq_len, n_heads]
        structure_compatibility = torch.zeros(batch_size, seq_len, seq_len, self.n_heads, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Compute compatibility between positions i and j
                # Use dot product of projected structure features
                compatibility = torch.sum(structure_proj[:, i, :] * structure_proj[:, j, :], dim=-1, keepdim=True)
                structure_compatibility[:, i, j, :] = compatibility
        
        # Transpose to [batch_size, n_heads, seq_len, seq_len]
        structure_bias = structure_compatibility.permute(0, 3, 1, 2)
        
        # Apply sigmoid to keep values in reasonable range
        structure_bias = torch.sigmoid(structure_bias) - 0.5  # Center around 0
        
        return structure_bias
    
    def _apply_mask(self, attention_scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply attention mask to attention scores.
        
        Args:
            attention_scores: Attention scores [batch_size, n_heads, seq_len, seq_len]
            mask: Attention mask
            
        Returns:
            Masked attention scores
        """
        if mask.dim() == 2:
            # Convert [batch_size, seq_len] to [batch_size, 1, 1, seq_len]
            mask = mask.unsqueeze(1).unsqueeze(2)
        elif mask.dim() == 3:
            # Convert [batch_size, seq_len, seq_len] to [batch_size, 1, seq_len, seq_len]
            mask = mask.unsqueeze(1)
        
        # Apply mask (0 means masked positions)
        attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        
        return attention_scores
    
    def get_interaction_matrix(self) -> torch.Tensor:
        """
        Get the learned amino acid interaction matrix.
        
        Returns:
            Symmetric interaction matrix [25, 25]
        """
        # Ensure symmetry
        return (self.interaction_matrix + self.interaction_matrix.T) / 2
    
    def get_physics_parameters(self) -> dict:
        """
        Get current physics parameter values.
        
        Returns:
            Dictionary of physics parameters
        """
        return {
            'physics_scale': self.physics_scale.item(),
            'distance_decay': self.distance_decay.item(),
            'distance_bias_norm': torch.norm(self.distance_bias).item(),
            'interaction_matrix_norm': torch.norm(self.interaction_matrix).item()
        }


class MultiHeadPhysicsAttention(nn.Module):
    """
    Multi-head physics-informed attention with residual connections and layer norm.
    
    This is a complete attention block that can be used as a drop-in replacement
    for standard transformer attention blocks.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        """
        Initialize multi-head physics attention block.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.attention = PhysicsInformedAttention(d_model, n_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                physics_features: Optional[torch.Tensor] = None,
                amino_acid_ids: Optional[torch.Tensor] = None,
                secondary_structure: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> torch.Tensor:
        """
        Forward pass with residual connection and layer normalization.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            physics_features: Optional physics features
            amino_acid_ids: Optional amino acid token IDs
            secondary_structure: Optional secondary structure one-hot
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Pre-layer norm
        normed_x = self.layer_norm(x)
        
        # Attention
        if return_attention:
            attn_output, attention_weights = self.attention(
                normed_x, physics_features, amino_acid_ids, secondary_structure, mask, return_attention=True
            )
        else:
            attn_output = self.attention(normed_x, physics_features, amino_acid_ids, secondary_structure, mask)
        
        # Residual connection
        output = x + self.dropout(attn_output)
        
        if return_attention:
            return output, attention_weights
        return output