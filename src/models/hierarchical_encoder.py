"""
Hierarchical encoder that processes protein information at multiple scales.

This module implements multi-scale processing at different hierarchical levels:
1. Amino acid level (residue-by-residue) - full resolution
2. Motif level (local structural patterns) - 2x downsampled  
3. Domain level (global protein structure) - 4x downsampled
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from src.models.physics_attention import MultiHeadPhysicsAttention


class HierarchicalEncoder(nn.Module):
    """
    Multi-scale protein encoder that processes information at different hierarchical levels.
    
    The encoder processes protein sequences through three hierarchical levels:
    - Level 1: Amino acid level (residue-by-residue processing)
    - Level 2: Motif level (local structural patterns, 2x downsampled)
    - Level 3: Domain level (global protein structure, 4x downsampled)
    
    Cross-level attention mechanisms enable information exchange between scales.
    """
    
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 n_layers: int, 
                 levels: int = 3,
                 dropout: float = 0.1,
                 use_physics_attention: bool = True):
        """
        Initialize hierarchical encoder.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Total number of transformer layers (distributed across levels)
            levels: Number of hierarchical levels
            dropout: Dropout probability
            use_physics_attention: Whether to use physics-informed attention
        """
        super().__init__()
        self.levels = levels
        self.d_model = d_model
        self.n_heads = n_heads
        self.use_physics_attention = use_physics_attention
        
        # Distribute layers across hierarchical levels
        layers_per_level = max(1, n_layers // levels)
        
        # Create transformer encoders for each hierarchical level
        self.level_encoders = nn.ModuleList()
        for level in range(levels):
            if use_physics_attention:
                # Use physics-informed attention blocks
                encoder_layers = nn.ModuleList([
                    MultiHeadPhysicsAttention(d_model, n_heads, dropout)
                    for _ in range(layers_per_level)
                ])
            else:
                # Use standard transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                )
                encoder_layers = nn.TransformerEncoder(encoder_layer, layers_per_level)
            
            self.level_encoders.append(encoder_layers)
        
        # Cross-level attention mechanisms for information integration
        self.cross_level_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            for _ in range(levels - 1)
        ])
        
        # Pooling strategies for different levels (downsampling)
        self.pooling_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=2, stride=2, padding=0)
            for i in range(levels - 1)
        ])
        
        # Upsampling layers for feature alignment
        self.upsampling_layers = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2, padding=0)
            for i in range(levels - 1)
        ])
        
        # Feature fusion layer
        self.fusion_layer = nn.Linear(d_model * levels, d_model)
        
        # Layer normalization for each level
        self.level_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(levels)
        ])
        
        # Final output normalization
        self.output_norm = nn.LayerNorm(d_model)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                physics_features: Optional[torch.Tensor] = None,
                amino_acid_ids: Optional[torch.Tensor] = None,
                secondary_structure: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process input through hierarchical levels and fuse information.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            physics_features: Optional physics features [batch_size, physics_dim]
            amino_acid_ids: Optional amino acid token IDs [batch_size, seq_len]
            secondary_structure: Optional secondary structure [batch_size, seq_len, 8]
            mask: Optional attention mask [batch_size, seq_len]
        
        Returns:
            Fused hierarchical features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store features from each level
        level_features = []
        current_features = x
        current_mask = mask
        current_aa_ids = amino_acid_ids
        current_ss = secondary_structure
        
        # Process each hierarchical level
        for level in range(self.levels):
            # Encode at current resolution
            if self.use_physics_attention:
                encoded = self._encode_with_physics_attention(
                    current_features, level, physics_features, 
                    current_aa_ids, current_ss, current_mask
                )
            else:
                encoded = self._encode_with_standard_attention(
                    current_features, level, current_mask
                )
            
            # Apply layer normalization
            encoded = self.level_norms[level](encoded)
            level_features.append(encoded)
            
            # Downsample for next level (except last level)
            if level < self.levels - 1:
                current_features, current_mask, current_aa_ids, current_ss = self._downsample_features(
                    encoded, current_mask, current_aa_ids, current_ss, level
                )
        
        # Cross-level attention for information integration
        integrated_features = self._integrate_cross_level_features(level_features, seq_len)
        
        # Concatenate and fuse all levels
        fused_features = self._fuse_hierarchical_features(integrated_features, seq_len)
        
        # Final normalization and dropout
        output = self.output_norm(fused_features)
        output = self.dropout(output)
        
        return output
    
    def _encode_with_physics_attention(self, 
                                     features: torch.Tensor,
                                     level: int,
                                     physics_features: Optional[torch.Tensor],
                                     amino_acid_ids: Optional[torch.Tensor],
                                     secondary_structure: Optional[torch.Tensor],
                                     mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode features using physics-informed attention."""
        encoded = features
        for attention_block in self.level_encoders[level]:
            encoded = attention_block(
                encoded,
                physics_features=physics_features,
                amino_acid_ids=amino_acid_ids,
                secondary_structure=secondary_structure,
                mask=mask
            )
        return encoded
    
    def _encode_with_standard_attention(self, 
                                      features: torch.Tensor,
                                      level: int,
                                      mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Encode features using standard transformer attention."""
        # Convert mask format for standard transformer
        if mask is not None:
            # Convert boolean mask to attention mask
            attention_mask = ~mask  # Invert: True means attend, False means mask
        else:
            attention_mask = None
        
        encoded = self.level_encoders[level](features, src_key_padding_mask=attention_mask)
        return encoded
    
    def _downsample_features(self, 
                           features: torch.Tensor,
                           mask: Optional[torch.Tensor],
                           amino_acid_ids: Optional[torch.Tensor],
                           secondary_structure: Optional[torch.Tensor],
                           level: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], 
                                              Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Downsample features and associated data for next hierarchical level."""
        # Transpose for conv1d: [batch_size, d_model, seq_len]
        features_transposed = features.transpose(1, 2)
        
        # Apply pooling
        downsampled = self.pooling_layers[level](features_transposed)
        
        # Transpose back: [batch_size, seq_len, d_model]
        downsampled = downsampled.transpose(1, 2)
        
        # Downsample mask if provided
        downsampled_mask = None
        if mask is not None:
            stride = 2
            downsampled_mask = mask[:, ::stride]
            # Ensure we don't go beyond the downsampled sequence length
            if downsampled_mask.shape[1] > downsampled.shape[1]:
                downsampled_mask = downsampled_mask[:, :downsampled.shape[1]]
        
        # Downsample amino acid IDs if provided
        downsampled_aa_ids = None
        if amino_acid_ids is not None:
            stride = 2
            downsampled_aa_ids = amino_acid_ids[:, ::stride]
            if downsampled_aa_ids.shape[1] > downsampled.shape[1]:
                downsampled_aa_ids = downsampled_aa_ids[:, :downsampled.shape[1]]
        
        # Downsample secondary structure if provided
        downsampled_ss = None
        if secondary_structure is not None:
            stride = 2
            downsampled_ss = secondary_structure[:, ::stride, :]
            if downsampled_ss.shape[1] > downsampled.shape[1]:
                downsampled_ss = downsampled_ss[:, :downsampled.shape[1], :]
        
        return downsampled, downsampled_mask, downsampled_aa_ids, downsampled_ss
    
    def _integrate_cross_level_features(self, 
                                      level_features: List[torch.Tensor],
                                      target_seq_len: int) -> List[torch.Tensor]:
        """Integrate features across hierarchical levels using cross-attention."""
        integrated_features = level_features.copy()
        
        # Apply cross-level attention from higher to lower levels
        for i in range(len(self.cross_level_attention)):
            query = integrated_features[i]  # Lower level (higher resolution)
            key_value = integrated_features[i + 1]  # Higher level (lower resolution)
            
            # Upsample higher level features to match query resolution
            if key_value.shape[1] != query.shape[1]:
                key_value = self._upsample_features(key_value, target_length=query.shape[1])
            
            # Apply cross-level attention
            attended, _ = self.cross_level_attention[i](
                query, key_value, key_value
            )
            
            # Residual connection
            integrated_features[i] = query + attended
        
        return integrated_features
    
    def _upsample_features(self, features: torch.Tensor, target_length: int) -> torch.Tensor:
        """Upsample features to target sequence length."""
        if features.shape[1] == target_length:
            return features
        
        # Use linear interpolation for upsampling
        # Transpose to [batch_size, d_model, seq_len] for interpolation
        features_transposed = features.transpose(1, 2)
        
        # Interpolate to target length
        upsampled = F.interpolate(
            features_transposed,
            size=target_length,
            mode='linear',
            align_corners=False
        )
        
        # Transpose back to [batch_size, seq_len, d_model]
        return upsampled.transpose(1, 2)
    
    def _fuse_hierarchical_features(self, 
                                  integrated_features: List[torch.Tensor],
                                  target_seq_len: int) -> torch.Tensor:
        """Fuse features from all hierarchical levels."""
        # Align all features to the same sequence length
        aligned_features = []
        for features in integrated_features:
            if features.shape[1] != target_seq_len:
                aligned = self._upsample_features(features, target_seq_len)
            else:
                aligned = features
            aligned_features.append(aligned)
        
        # Concatenate features from all levels
        concatenated = torch.cat(aligned_features, dim=-1)
        
        # Fuse through linear projection
        fused = self.fusion_layer(concatenated)
        
        return fused
    
    def get_level_features(self, 
                          x: torch.Tensor,
                          physics_features: Optional[torch.Tensor] = None,
                          amino_acid_ids: Optional[torch.Tensor] = None,
                          secondary_structure: Optional[torch.Tensor] = None,
                          mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get features from each hierarchical level separately (for analysis/visualization).
        
        Returns:
            List of feature tensors, one for each hierarchical level
        """
        batch_size, seq_len, d_model = x.shape
        level_features = []
        current_features = x
        current_mask = mask
        current_aa_ids = amino_acid_ids
        current_ss = secondary_structure
        
        # Process each hierarchical level
        for level in range(self.levels):
            # Encode at current resolution
            if self.use_physics_attention:
                encoded = self._encode_with_physics_attention(
                    current_features, level, physics_features, 
                    current_aa_ids, current_ss, current_mask
                )
            else:
                encoded = self._encode_with_standard_attention(
                    current_features, level, current_mask
                )
            
            # Apply layer normalization
            encoded = self.level_norms[level](encoded)
            level_features.append(encoded)
            
            # Downsample for next level (except last level)
            if level < self.levels - 1:
                current_features, current_mask, current_aa_ids, current_ss = self._downsample_features(
                    encoded, current_mask, current_aa_ids, current_ss, level
                )
        
        return level_features
    
    def get_cross_level_attention_weights(self, 
                                        x: torch.Tensor,
                                        physics_features: Optional[torch.Tensor] = None,
                                        amino_acid_ids: Optional[torch.Tensor] = None,
                                        secondary_structure: Optional[torch.Tensor] = None,
                                        mask: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Get cross-level attention weights for interpretability.
        
        Returns:
            List of attention weight tensors between hierarchical levels
        """
        level_features = self.get_level_features(x, physics_features, amino_acid_ids, secondary_structure, mask)
        attention_weights = []
        
        # Compute cross-level attention weights
        for i in range(len(self.cross_level_attention)):
            query = level_features[i]
            key_value = level_features[i + 1]
            
            # Upsample if needed
            if key_value.shape[1] != query.shape[1]:
                key_value = self._upsample_features(key_value, target_length=query.shape[1])
            
            # Get attention weights
            _, weights = self.cross_level_attention[i](query, key_value, key_value)
            attention_weights.append(weights)
        
        return attention_weights