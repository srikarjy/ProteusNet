"""
Unit tests for hierarchical encoder architecture.
"""
import pytest
import torch
import torch.nn as nn
from src.models.hierarchical_encoder import HierarchicalEncoder


class TestHierarchicalEncoder:
    """Test cases for HierarchicalEncoder class."""
    
    def test_initialization_default(self):
        """Test default hierarchical encoder initialization."""
        d_model, n_heads, n_layers = 256, 8, 12
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        assert encoder.levels == 3
        assert encoder.d_model == d_model
        assert encoder.n_heads == n_heads
        assert encoder.use_physics_attention == True
        assert len(encoder.level_encoders) == 3
        assert len(encoder.cross_level_attention) == 2
        assert len(encoder.pooling_layers) == 2
    
    def test_initialization_custom(self):
        """Test custom hierarchical encoder initialization."""
        d_model, n_heads, n_layers, levels = 128, 4, 8, 4
        encoder = HierarchicalEncoder(
            d_model, n_heads, n_layers, levels=levels, 
            dropout=0.2, use_physics_attention=False
        )
        
        assert encoder.levels == levels
        assert encoder.use_physics_attention == False
        assert len(encoder.level_encoders) == levels
        assert len(encoder.cross_level_attention) == levels - 1
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 32, 256, 8, 12
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_forward_with_physics_features(self):
        """Test forward pass with physics features."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 8
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, use_physics_attention=True)
        
        x = torch.randn(batch_size, seq_len, d_model)
        # Create physics features
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        secondary_structure = torch.rand(batch_size, seq_len, 8)
        
        output = encoder(x, 
                        physics_features=physics_features,
                        amino_acid_ids=amino_acid_ids,
                        secondary_structure=secondary_structure)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_forward_with_mask(self):
        """Test forward pass with attention mask."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 24, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        x = torch.randn(batch_size, seq_len, d_model)
        # Create attention mask
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        mask[:, -8:] = False  # Mask last 8 positions
        
        output = encoder(x, mask=mask)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_forward_standard_attention(self):
        """Test forward pass with standard attention."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, use_physics_attention=False)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = encoder(x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_different_sequence_lengths(self):
        """Test encoder with different sequence lengths."""
        d_model, n_heads, n_layers = 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        for seq_len in [8, 16, 32, 64]:
            batch_size = 2
            x = torch.randn(batch_size, seq_len, d_model)
            output = encoder(x)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_different_hierarchical_levels(self):
        """Test encoder with different numbers of hierarchical levels."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 32, 128, 4, 12
        
        for levels in [2, 3, 4, 5]:
            encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=levels)
            x = torch.randn(batch_size, seq_len, d_model)
            output = encoder(x)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert len(encoder.level_encoders) == levels
            assert len(encoder.cross_level_attention) == levels - 1
    
    def test_downsampling_features(self):
        """Test feature downsampling functionality."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 32, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        # Test downsampling
        features = torch.randn(batch_size, seq_len, d_model)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        secondary_structure = torch.rand(batch_size, seq_len, 8)
        
        # Test level 0 downsampling (stride 2)
        down_features, down_mask, down_aa, down_ss = encoder._downsample_features(
            features, mask, amino_acid_ids, secondary_structure, level=0
        )
        
        expected_len = seq_len // 2
        assert down_features.shape == (batch_size, expected_len, d_model)
        assert down_mask.shape == (batch_size, expected_len)
        assert down_aa.shape == (batch_size, expected_len)
        assert down_ss.shape == (batch_size, expected_len, 8)
    
    def test_upsampling_features(self):
        """Test feature upsampling functionality."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        # Create downsampled features
        downsampled_features = torch.randn(batch_size, seq_len // 2, d_model)
        
        # Upsample back to original length
        upsampled = encoder._upsample_features(downsampled_features, target_length=seq_len)
        
        assert upsampled.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(upsampled).any()
    
    def test_get_level_features(self):
        """Test getting features from each hierarchical level."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=3)
        
        x = torch.randn(batch_size, seq_len, d_model)
        level_features = encoder.get_level_features(x)
        
        assert len(level_features) == 3
        # Check shapes for each level
        assert level_features[0].shape == (batch_size, seq_len, d_model)  # Level 0: full resolution
        assert level_features[1].shape == (batch_size, seq_len // 2, d_model)  # Level 1: 2x downsampled
        assert level_features[2].shape == (batch_size, seq_len // 4, d_model)  # Level 2: 4x downsampled
    
    def test_get_cross_level_attention_weights(self):
        """Test getting cross-level attention weights."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=3)
        
        x = torch.randn(batch_size, seq_len, d_model)
        attention_weights = encoder.get_cross_level_attention_weights(x)
        
        assert len(attention_weights) == 2  # 3 levels - 1 = 2 cross-level attentions
        # Check attention weight shapes
        for weights in attention_weights:
            assert weights.shape[0] == batch_size
            # Attention weights should be square matrices
            assert weights.shape[1] == weights.shape[2]
    
    def test_gradient_flow(self):
        """Test gradient flow through hierarchical encoder."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        physics_features = torch.rand(batch_size, seq_len * seq_len + 10, requires_grad=True)
        
        output = encoder(x, physics_features=physics_features)
        loss = output.sum()
        loss.backward()
        
        # Check that input has gradients
        assert x.grad is not None
        assert physics_features.grad is not None
        
        # Check that most model parameters have gradients
        params_with_grad = 0
        total_params = 0
        for param in encoder.parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
        
        # At least 80% of parameters should have gradients
        assert params_with_grad / total_params > 0.8
    
    def test_cross_level_integration(self):
        """Test cross-level feature integration."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=3)
        
        # Create mock level features
        level_features = [
            torch.randn(batch_size, seq_len, d_model),      # Level 0: full resolution
            torch.randn(batch_size, seq_len // 2, d_model), # Level 1: 2x downsampled
            torch.randn(batch_size, seq_len // 4, d_model)  # Level 2: 4x downsampled
        ]
        
        integrated = encoder._integrate_cross_level_features(level_features, seq_len)
        
        assert len(integrated) == 3
        # All integrated features should maintain their original shapes
        assert integrated[0].shape == (batch_size, seq_len, d_model)
        assert integrated[1].shape == (batch_size, seq_len // 2, d_model)
        assert integrated[2].shape == (batch_size, seq_len // 4, d_model)
    
    def test_feature_fusion(self):
        """Test hierarchical feature fusion."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=3)
        
        # Create mock integrated features
        integrated_features = [
            torch.randn(batch_size, seq_len, d_model),      # Level 0
            torch.randn(batch_size, seq_len // 2, d_model), # Level 1
            torch.randn(batch_size, seq_len // 4, d_model)  # Level 2
        ]
        
        fused = encoder._fuse_hierarchical_features(integrated_features, seq_len)
        
        assert fused.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(fused).any()
    
    def test_memory_efficiency(self):
        """Test memory efficiency with different configurations."""
        batch_size, seq_len, d_model = 1, 64, 256
        
        # Test with different numbers of layers and levels
        configs = [
            (4, 6, 2),   # n_heads, n_layers, levels
            (8, 12, 3),
            (4, 8, 4),
        ]
        
        for n_heads, n_layers, levels in configs:
            encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=levels)
            x = torch.randn(batch_size, seq_len, d_model)
            
            # Should not raise memory errors
            output = encoder(x)
            assert output.shape == (batch_size, seq_len, d_model)
    
    def test_deterministic_output(self):
        """Test that output is deterministic given same input."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        encoder.eval()  # Set to evaluation mode
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Run forward pass twice
        output1 = encoder(x)
        output2 = encoder(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_training_vs_eval_mode(self):
        """Test behavior difference between training and evaluation modes."""
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 16, 128, 4, 6
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, dropout=0.5)
        
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Training mode
        encoder.train()
        output_train1 = encoder(x)
        output_train2 = encoder(x)
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output_train1, output_train2, atol=1e-6)
        
        # Evaluation mode
        encoder.eval()
        output_eval1 = encoder(x)
        output_eval2 = encoder(x)
        
        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2, atol=1e-6)


class TestHierarchicalEncoderIntegration:
    """Integration tests for hierarchical encoder with other components."""
    
    def test_integration_with_physics_attention(self):
        """Test integration with physics-informed attention."""
        from src.data.protein_tokenizer import PhysicsFeatureExtractor
        
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 32, 256, 8, 12
        
        # Create hierarchical encoder with physics attention
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, use_physics_attention=True)
        
        # Create realistic physics features
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
        
        # Test with real physics features
        x = torch.randn(batch_size, seq_len, d_model)
        amino_acid_ids = torch.randint(0, 25, (batch_size, seq_len))
        
        output = encoder(x, 
                        physics_features=physics_features,
                        amino_acid_ids=amino_acid_ids)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_integration_with_protein_batch(self):
        """Test integration with protein batch data structures."""
        from src.data.protein_batch import ProteinBatch
        
        batch_size, seq_len, d_model, n_heads, n_layers = 2, 24, 128, 4, 8
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers)
        
        # Create protein batch
        sequences = torch.randint(0, 25, (batch_size, seq_len))
        masks = torch.ones(batch_size, seq_len, dtype=torch.bool)
        masks[:, -4:] = False  # Mask last 4 positions
        
        batch = ProteinBatch(sequences=sequences, masks=masks)
        
        # Create input embeddings (would normally come from embedding layer)
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = encoder(x, mask=batch.masks)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_scalability(self):
        """Test scalability with larger inputs."""
        d_model, n_heads, n_layers = 512, 8, 12
        encoder = HierarchicalEncoder(d_model, n_heads, n_layers, levels=3)
        
        # Test with larger sequence lengths
        for seq_len in [64, 128, 256]:
            batch_size = 1  # Use smaller batch for memory efficiency
            x = torch.randn(batch_size, seq_len, d_model)
            output = encoder(x)
            assert output.shape == (batch_size, seq_len, d_model)


if __name__ == "__main__":
    pytest.main([__file__])