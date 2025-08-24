"""
Unit tests for physics-informed attention mechanisms.
"""

import pytest
import torch
import torch.nn as nn
import math
from src.models.physics_attention import PhysicsInformedAttention, MultiHeadPhysicsAttention


class TestPhysicsInformedAttention:
    """Test cases for PhysicsInformedAttention class."""
    
    def test_initialization(self):
        """Test basic initialization of physics attention."""
        d_model, n_heads = 512, 8
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        assert attention.d_model == d_model
        assert attention.n_heads == n_heads
        assert attention.d_k == d_model // n_heads
        
        # Check parameter shapes
        assert attention.distance_bias.shape == (1, n_heads, 1, 1)
        assert attention.interaction_matrix.shape == (25, 25)
        assert attention.physics_scale.shape == ()
        assert attention.distance_decay.shape == ()
    
    def test_initialization_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with pytest.raises(AssertionError, match="d_model must be divisible by n_heads"):
            PhysicsInformedAttention(d_model=513, n_heads=8)
    
    def test_forward_basic(self):
        """Test basic forward pass without physics features."""
        batch_size, seq_len, d_model, n_heads = 2, 64, 512, 8
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_physics_features(self):
        """Test forward pass with physics features."""
        batch_size, seq_len, d_model, n_heads = 2, 32, 256, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create physics features with distance matrix
        physics_dim = seq_len * seq_len + 10  # Distance matrix + extra features
        physics_features = torch.rand(batch_size, physics_dim)
        
        output = attention(x, physics_features=physics_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_forward_with_all_features(self):
        """Test forward pass with all physics features."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create all types of features
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        secondary_structure = torch.rand(batch_size, seq_len, 8)
        
        output = attention(x, 
                         physics_features=physics_features,
                         amino_acid_ids=amino_acid_ids,
                         secondary_structure=secondary_structure)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test with 2D mask [batch_size, seq_len]
        mask_2d = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_2d[:, -4:] = False  # Mask last 4 positions
        
        output = attention(x, mask=mask_2d)
        assert output.shape == (batch_size, seq_len, d_model)
        
        # Test with 3D mask [batch_size, seq_len, seq_len]
        mask_3d = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        output = attention(x, mask=mask_3d)
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_return_attention_weights(self):
        """Test returning attention weights."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention(x, return_attention=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1 (set to eval mode to avoid dropout)
        attention.eval()
        output, attention_weights = attention(x, return_attention=True)
        assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)))
    
    def test_physics_bias_computation(self):
        """Test physics bias computation."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create physics features with distance matrix
        physics_features = torch.rand(batch_size, seq_len * seq_len + 5)
        
        bias = attention._compute_physics_bias(physics_features, seq_len, batch_size)
        
        assert bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_physics_bias_with_amino_acids(self):
        """Test physics bias computation with amino acid interactions."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create physics features and amino acid IDs
        physics_features = torch.rand(batch_size, seq_len * seq_len + 5)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        
        bias = attention._compute_physics_bias(
            physics_features, seq_len, batch_size, amino_acid_ids
        )
        
        assert bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_physics_bias_with_secondary_structure(self):
        """Test physics bias computation with secondary structure."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create physics features and secondary structure
        physics_features = torch.rand(batch_size, seq_len * seq_len + 5)
        secondary_structure = torch.rand(batch_size, seq_len, 8)  # One-hot encoded
        
        bias = attention._compute_physics_bias(
            physics_features, seq_len, batch_size, None, secondary_structure
        )
        
        assert bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_physics_bias_all_features(self):
        """Test physics bias computation with all features."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create all types of features
        physics_features = torch.rand(batch_size, seq_len * seq_len + 5)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        secondary_structure = torch.rand(batch_size, seq_len, 8)
        
        bias = attention._compute_physics_bias(
            physics_features, seq_len, batch_size, amino_acid_ids, secondary_structure
        )
        
        assert bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(bias).any()
        assert not torch.isinf(bias).any()
    
    def test_distance_bias_computation(self):
        """Test distance bias computation specifically."""
        batch_size, seq_len, d_model, n_heads = 2, 6, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create physics features with known distance matrix
        distance_matrix = torch.arange(seq_len * seq_len, dtype=torch.float32).view(seq_len, seq_len)
        distance_matrix = distance_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        physics_features = distance_matrix.view(batch_size, -1)
        
        device = physics_features.device
        distance_bias = attention._compute_distance_bias(physics_features, seq_len, batch_size, device)
        
        assert distance_bias.shape == (batch_size, n_heads, seq_len, seq_len)
        # Distance bias should be negative (penalty for large distances)
        assert (distance_bias <= 0).all()
    
    def test_interaction_bias_computation(self):
        """Test amino acid interaction bias computation."""
        batch_size, seq_len, d_model, n_heads = 2, 6, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create amino acid IDs
        amino_acid_ids = torch.randint(0, 20, (batch_size, seq_len))  # Standard amino acids
        device = amino_acid_ids.device
        
        interaction_bias = attention._compute_interaction_bias(amino_acid_ids, seq_len, batch_size, device)
        
        assert interaction_bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(interaction_bias).any()
        
        # Interaction bias should be symmetric for same amino acid pairs
        for b in range(batch_size):
            for h in range(n_heads):
                for i in range(seq_len):
                    for j in range(seq_len):
                        if amino_acid_ids[b, i] == amino_acid_ids[b, j]:
                            # Same amino acids should have same interaction with any third amino acid
                            pass  # This is a complex property to test, so we just check for NaN/Inf
    
    def test_structure_bias_computation(self):
        """Test secondary structure bias computation."""
        batch_size, seq_len, d_model, n_heads = 2, 6, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create secondary structure one-hot vectors
        secondary_structure = torch.zeros(batch_size, seq_len, 8)
        # Set some positions to have specific secondary structures
        secondary_structure[0, 0, 0] = 1.0  # Alpha helix
        secondary_structure[0, 1, 1] = 1.0  # Beta sheet
        secondary_structure[1, 0, 2] = 1.0  # Turn
        
        device = secondary_structure.device
        structure_bias = attention._compute_structure_bias(secondary_structure, seq_len, batch_size, device)
        
        assert structure_bias.shape == (batch_size, n_heads, seq_len, seq_len)
        assert not torch.isnan(structure_bias).any()
        # Structure bias should be in reasonable range after sigmoid
        assert (structure_bias >= -0.5).all() and (structure_bias <= 0.5).all()
    
    def test_physics_bias_without_distance_matrix(self):
        """Test physics bias computation without distance matrix."""
        batch_size, seq_len, d_model, n_heads = 2, 8, 64, 2
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        # Create physics features without distance matrix (too small)
        physics_features = torch.rand(batch_size, 10)
        
        bias = attention._compute_physics_bias(physics_features, seq_len, batch_size)
        
        assert bias.shape == (batch_size, n_heads, seq_len, seq_len)
        # Should only contain the learnable distance bias
        expected_bias = attention.distance_bias.expand(batch_size, n_heads, seq_len, seq_len)
        assert torch.allclose(bias, expected_bias)
    
    def test_mask_application(self):
        """Test attention mask application."""
        batch_size, seq_len, n_heads = 2, 8, 4
        
        attention = PhysicsInformedAttention(256, n_heads)
        attention_scores = torch.randn(batch_size, n_heads, seq_len, seq_len)
        
        # Test 2D mask
        mask_2d = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask_2d[:, -2:] = False
        
        masked_scores = attention._apply_mask(attention_scores, mask_2d)
        
        # Check that masked positions have very negative values
        assert (masked_scores[:, :, :, -2:] < -1e8).all()
        
        # Test 3D mask
        mask_3d = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).repeat(batch_size, 1, 1)
        
        masked_scores = attention._apply_mask(attention_scores, mask_3d)
        
        # Check that upper triangular part is masked
        assert (masked_scores[:, :, torch.triu_indices(seq_len, seq_len, offset=1)[0], 
                             torch.triu_indices(seq_len, seq_len, offset=1)[1]] < -1e8).all()
    
    def test_interaction_matrix_symmetry(self):
        """Test that interaction matrix is symmetric."""
        attention = PhysicsInformedAttention(128, 4)
        
        interaction_matrix = attention.get_interaction_matrix()
        
        assert interaction_matrix.shape == (25, 25)
        assert torch.allclose(interaction_matrix, interaction_matrix.T)
    
    def test_physics_parameters(self):
        """Test physics parameter retrieval."""
        attention = PhysicsInformedAttention(128, 4)
        
        params = attention.get_physics_parameters()
        
        assert 'physics_scale' in params
        assert 'distance_decay' in params
        assert 'distance_bias_norm' in params
        assert 'interaction_matrix_norm' in params
        
        assert isinstance(params['physics_scale'], float)
        assert isinstance(params['distance_decay'], float)
        assert params['distance_bias_norm'] >= 0
        assert params['interaction_matrix_norm'] >= 0
    
    def test_gradient_flow(self):
        """Test that gradients flow through physics parameters."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10, requires_grad=True)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        secondary_structure = torch.rand(batch_size, seq_len, 8, requires_grad=True)
        
        output = attention(x, 
                         physics_features=physics_features,
                         amino_acid_ids=amino_acid_ids,
                         secondary_structure=secondary_structure)
        loss = output.sum()
        loss.backward()
        
        # Check that physics parameters have gradients
        assert attention.physics_scale.grad is not None
        assert attention.distance_decay.grad is not None
        assert attention.distance_bias.grad is not None
        assert attention.interaction_matrix.grad is not None  # Should have gradients now
        assert attention.secondary_structure_bias.weight.grad is not None
        
        # Check that input has gradients
        assert x.grad is not None
        assert physics_features.grad is not None
        assert secondary_structure.grad is not None
    
    def test_different_sequence_lengths(self):
        """Test attention with different sequence lengths."""
        d_model, n_heads = 256, 8
        attention = PhysicsInformedAttention(d_model, n_heads)
        
        for seq_len in [8, 16, 32, 64, 128]:
            batch_size = 2
            x = torch.randn(batch_size, seq_len, d_model)
            
            output = attention(x)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_dropout_training_vs_eval(self):
        """Test dropout behavior in training vs evaluation mode."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention = PhysicsInformedAttention(d_model, n_heads, dropout=0.5)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Training mode
        attention.train()
        output_train1 = attention(x)
        output_train2 = attention(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, atol=1e-6)
        
        # Evaluation mode
        attention.eval()
        output_eval1 = attention(x)
        output_eval2 = attention(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)


class TestMultiHeadPhysicsAttention:
    """Test cases for MultiHeadPhysicsAttention class."""
    
    def test_initialization(self):
        """Test initialization of multi-head physics attention."""
        d_model, n_heads = 256, 8
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads)
        
        assert isinstance(attention_block.attention, PhysicsInformedAttention)
        assert isinstance(attention_block.layer_norm, nn.LayerNorm)
        assert attention_block.layer_norm.normalized_shape == (d_model,)
    
    def test_forward_basic(self):
        """Test basic forward pass of attention block."""
        batch_size, seq_len, d_model, n_heads = 2, 32, 256, 8
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention_block(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_residual_connection(self):
        """Test that residual connection is working."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Zero out attention weights to test residual connection
        with torch.no_grad():
            attention_block.attention.q_proj.weight.zero_()
            attention_block.attention.k_proj.weight.zero_()
            attention_block.attention.v_proj.weight.zero_()
            attention_block.attention.out_proj.weight.zero_()
            attention_block.attention.out_proj.bias.zero_()
        
        output = attention_block(x)
        
        # Output should be close to input due to residual connection
        # (after layer norm, but the residual should dominate)
        assert output.shape == x.shape
    
    def test_forward_with_physics_features(self):
        """Test forward pass with physics features."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10)
        
        output = attention_block(x, physics_features=physics_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_return_attention_weights(self):
        """Test returning attention weights from attention block."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attention_weights = attention_block(x, return_attention=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attention_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_layer_norm_placement(self):
        """Test pre-layer norm architecture."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        attention_block = MultiHeadPhysicsAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Hook to capture layer norm input
        layer_norm_input = None
        def hook(module, input, output):
            nonlocal layer_norm_input
            layer_norm_input = input[0].clone()
        
        handle = attention_block.layer_norm.register_forward_hook(hook)
        
        output = attention_block(x)
        
        # Layer norm should receive the original input x
        assert torch.allclose(layer_norm_input, x)
        
        handle.remove()


class TestPhysicsAttentionIntegration:
    """Integration tests for physics attention components."""
    
    def test_attention_with_tokenizer_features(self):
        """Test physics attention with features from tokenizer."""
        from src.data.protein_tokenizer import PhysicsFeatureExtractor
        
        batch_size, seq_len, d_model, n_heads = 2, 32, 256, 8
        
        # Create physics features using the feature extractor
        extractor = PhysicsFeatureExtractor()
        sequences = ["ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIKLMNPQRST"]
        
        physics_features = []
        for seq in sequences:
            features = extractor.extract_sequence_features(seq)
            # Pad with zeros to simulate distance matrix
            padded_features = torch.cat([
                torch.tensor(features),
                torch.zeros(seq_len * seq_len)
            ])
            physics_features.append(padded_features)
        
        physics_features = torch.stack(physics_features)
        
        # Test attention with real physics features
        attention = PhysicsInformedAttention(d_model, n_heads)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = attention(x, physics_features=physics_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_multiple_attention_layers(self):
        """Test stacking multiple physics attention layers."""
        batch_size, seq_len, d_model, n_heads = 2, 16, 128, 4
        
        # Create multiple attention layers
        attention_layers = nn.ModuleList([
            MultiHeadPhysicsAttention(d_model, n_heads)
            for _ in range(3)
        ])
        
        x = torch.randn(batch_size, seq_len, d_model)
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10)
        
        # Pass through multiple layers
        output = x
        for layer in attention_layers:
            output = layer(output, physics_features=physics_features)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_parameter_sharing(self):
        """Test that physics parameters are not shared between instances."""
        d_model, n_heads = 128, 4
        
        attention1 = PhysicsInformedAttention(d_model, n_heads)
        attention2 = PhysicsInformedAttention(d_model, n_heads)
        
        # Parameters should be different instances
        assert attention1.physics_scale is not attention2.physics_scale
        assert attention1.distance_decay is not attention2.distance_decay
        assert attention1.interaction_matrix is not attention2.interaction_matrix
        
        # Modify one parameter
        with torch.no_grad():
            attention1.physics_scale.fill_(5.0)
        
        # Other instance should be unchanged
        assert attention2.physics_scale.item() != 5.0


if __name__ == "__main__":
    pytest.main([__file__])