"""
Unit tests for protein batch data structures and validation.
"""

import pytest
import torch
import numpy as np
from src.data.protein_batch import ProteinBatch, BatchConstructor


class TestProteinBatch:
    """Test cases for ProteinBatch class."""
    
    def test_basic_initialization(self):
        """Test basic ProteinBatch initialization."""
        batch_size, seq_length = 4, 128
        
        sequences = torch.randint(0, 25, (batch_size, seq_length))
        masks = torch.ones((batch_size, seq_length), dtype=torch.bool)
        
        batch = ProteinBatch(sequences=sequences, masks=masks)
        
        assert batch.batch_size == batch_size
        assert batch.seq_length == seq_length
        assert torch.equal(batch.sequences, sequences)
        assert torch.equal(batch.masks, masks)
    
    def test_full_initialization(self):
        """Test ProteinBatch initialization with all fields."""
        batch_size, seq_length = 2, 64
        num_functions = 1000
        physics_dim = 128
        evo_dim = 256
        
        sequences = torch.randint(0, 25, (batch_size, seq_length))
        masks = torch.ones((batch_size, seq_length), dtype=torch.bool)
        function_labels = torch.rand((batch_size, num_functions))
        structure_labels = torch.randint(0, 8, (batch_size, seq_length))
        stability_labels = torch.rand((batch_size, 1))
        physics_features = torch.rand((batch_size, physics_dim))
        evolutionary_features = torch.rand((batch_size, evo_dim))
        sequence_lengths = torch.tensor([60, 64])
        protein_ids = ["protein_1", "protein_2"]
        
        batch = ProteinBatch(
            sequences=sequences,
            masks=masks,
            function_labels=function_labels,
            structure_labels=structure_labels,
            stability_labels=stability_labels,
            physics_features=physics_features,
            evolutionary_features=evolutionary_features,
            sequence_lengths=sequence_lengths,
            protein_ids=protein_ids
        )
        
        assert batch.batch_size == batch_size
        assert batch.seq_length == seq_length
        assert torch.equal(batch.function_labels, function_labels)
        assert torch.equal(batch.structure_labels, structure_labels)
        assert torch.equal(batch.stability_labels, stability_labels)
        assert torch.equal(batch.physics_features, physics_features)
        assert torch.equal(batch.evolutionary_features, evolutionary_features)
        assert torch.equal(batch.sequence_lengths, sequence_lengths)
        assert batch.protein_ids == protein_ids
    
    def test_validation_invalid_sequences_dtype(self):
        """Test validation fails with invalid sequences dtype."""
        sequences = torch.rand((2, 64))  # Should be integer
        masks = torch.ones((2, 64), dtype=torch.bool)
        
        with pytest.raises(ValueError, match="sequences must have integer dtype"):
            ProteinBatch(sequences=sequences, masks=masks)
    
    def test_validation_mismatched_shapes(self):
        """Test validation fails with mismatched tensor shapes."""
        sequences = torch.randint(0, 25, (2, 64))
        masks = torch.ones((2, 32), dtype=torch.bool)  # Wrong shape
        
        with pytest.raises(ValueError, match="masks shape .* doesn't match sequences shape"):
            ProteinBatch(sequences=sequences, masks=masks)
    
    def test_validation_invalid_function_labels_batch_size(self):
        """Test validation fails with wrong function labels batch size."""
        sequences = torch.randint(0, 25, (2, 64))
        masks = torch.ones((2, 64), dtype=torch.bool)
        function_labels = torch.rand((3, 1000))  # Wrong batch size
        
        with pytest.raises(ValueError, match="function_labels batch size .* doesn't match"):
            ProteinBatch(sequences=sequences, masks=masks, function_labels=function_labels)
    
    def test_validation_invalid_structure_labels_shape(self):
        """Test validation fails with wrong structure labels shape."""
        sequences = torch.randint(0, 25, (2, 64))
        masks = torch.ones((2, 64), dtype=torch.bool)
        structure_labels = torch.randint(0, 8, (2, 32))  # Wrong sequence length
        
        with pytest.raises(ValueError, match="structure_labels shape .* doesn't match sequences shape"):
            ProteinBatch(sequences=sequences, masks=masks, structure_labels=structure_labels)
    
    def test_validation_invalid_sequence_lengths(self):
        """Test validation fails with invalid sequence lengths."""
        sequences = torch.randint(0, 25, (2, 64))
        masks = torch.ones((2, 64), dtype=torch.bool)
        sequence_lengths = torch.tensor([70, 80])  # Exceed max length
        
        with pytest.raises(ValueError, match="sequence_lengths cannot exceed max sequence length"):
            ProteinBatch(sequences=sequences, masks=masks, sequence_lengths=sequence_lengths)
    
    def test_to_device(self):
        """Test moving batch to different device."""
        sequences = torch.randint(0, 25, (2, 64))
        masks = torch.ones((2, 64), dtype=torch.bool)
        function_labels = torch.rand((2, 1000))
        
        batch = ProteinBatch(sequences=sequences, masks=masks, function_labels=function_labels)
        
        # Test moving to CPU (should work regardless of CUDA availability)
        cpu_batch = batch.to(torch.device('cpu'))
        assert cpu_batch.sequences.device == torch.device('cpu')
        assert cpu_batch.masks.device == torch.device('cpu')
        assert cpu_batch.function_labels.device == torch.device('cpu')
    
    def test_pin_memory(self):
        """Test pinning memory for batch tensors."""
        # Force tensors to be on CPU for this test
        sequences = torch.randint(0, 25, (2, 64)).cpu()
        masks = torch.ones((2, 64), dtype=torch.bool).cpu()
        
        batch = ProteinBatch(sequences=sequences, masks=masks)
        
        try:
            pinned_batch = batch.pin_memory()
            # Only check if pinning succeeded (may fail on some systems)
            if pinned_batch.sequences.is_pinned():
                assert pinned_batch.sequences.is_pinned()
                assert pinned_batch.masks.is_pinned()
        except RuntimeError:
            # Pin memory may not be available on all systems
            pytest.skip("Pin memory not available on this system")


class TestBatchConstructor:
    """Test cases for BatchConstructor class."""
    
    def test_initialization(self):
        """Test BatchConstructor initialization."""
        constructor = BatchConstructor(pad_token_id=0, max_length=512)
        assert constructor.pad_token_id == 0
        assert constructor.max_length == 512
    
    def test_pad_sequence_truncate(self):
        """Test sequence padding with truncation."""
        constructor = BatchConstructor(pad_token_id=0, max_length=10)
        sequence = list(range(15))  # Length 15, should be truncated
        
        padded_seq, mask = constructor.pad_sequence(sequence, target_length=10)
        
        assert len(padded_seq) == 10
        assert len(mask) == 10
        assert padded_seq == list(range(10))
        assert all(mask)
    
    def test_pad_sequence_pad(self):
        """Test sequence padding with actual padding."""
        constructor = BatchConstructor(pad_token_id=99, max_length=10)
        sequence = [1, 2, 3, 4, 5]  # Length 5, should be padded
        
        padded_seq, mask = constructor.pad_sequence(sequence, target_length=10)
        
        assert len(padded_seq) == 10
        assert len(mask) == 10
        assert padded_seq == [1, 2, 3, 4, 5, 99, 99, 99, 99, 99]
        assert mask == [True, True, True, True, True, False, False, False, False, False]
    
    def test_collate_fn_basic(self):
        """Test basic collate function."""
        constructor = BatchConstructor(pad_token_id=0, max_length=10)
        
        samples = [
            {'sequence': [1, 2, 3, 4]},
            {'sequence': [5, 6, 7]},
            {'sequence': [8, 9, 10, 11, 12]}
        ]
        
        batch = constructor.collate_fn(samples)
        
        assert batch.batch_size == 3
        assert batch.seq_length == 5  # Max length in samples
        assert torch.equal(batch.sequences[0], torch.tensor([1, 2, 3, 4, 0]))
        assert torch.equal(batch.sequences[1], torch.tensor([5, 6, 7, 0, 0]))
        assert torch.equal(batch.sequences[2], torch.tensor([8, 9, 10, 11, 12]))
        assert torch.equal(batch.sequence_lengths, torch.tensor([4, 3, 5]))
    
    def test_collate_fn_with_labels(self):
        """Test collate function with labels."""
        constructor = BatchConstructor(pad_token_id=0, max_length=10)
        
        samples = [
            {
                'sequence': [1, 2, 3],
                'function_labels': [0.1, 0.2, 0.3],
                'structure_labels': [1, 2, 3],
                'stability_labels': 0.5,
                'protein_id': 'prot1'
            },
            {
                'sequence': [4, 5],
                'function_labels': [0.4, 0.5, 0.6],
                'structure_labels': [4, 5],
                'stability_labels': 0.7,
                'protein_id': 'prot2'
            }
        ]
        
        batch = constructor.collate_fn(samples)
        
        assert batch.batch_size == 2
        assert batch.function_labels is not None
        assert batch.structure_labels is not None
        assert batch.stability_labels is not None
        assert batch.protein_ids == ['prot1', 'prot2']
        
        # Check function labels
        expected_func = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        assert torch.allclose(batch.function_labels, expected_func)
        
        # Check stability labels
        expected_stab = torch.tensor([[0.5], [0.7]])
        assert torch.allclose(batch.stability_labels, expected_stab)
    
    def test_create_batch_from_sequences(self):
        """Test creating batch from sequences."""
        constructor = BatchConstructor(pad_token_id=0, max_length=10)
        
        sequences = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9]
        ]
        
        function_labels = [
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]
        
        batch = constructor.create_batch_from_sequences(
            sequences, 
            function_labels=function_labels
        )
        
        assert batch.batch_size == 3
        assert batch.function_labels is not None
        assert torch.allclose(batch.function_labels, torch.tensor(function_labels))
    
    def test_collate_fn_max_length_limit(self):
        """Test that collate function respects max_length limit."""
        constructor = BatchConstructor(pad_token_id=0, max_length=3)
        
        samples = [
            {'sequence': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},  # Long sequence
            {'sequence': [11, 12]}
        ]
        
        batch = constructor.collate_fn(samples)
        
        assert batch.seq_length == 3  # Should be limited by max_length
        assert torch.equal(batch.sequences[0], torch.tensor([1, 2, 3]))
        assert torch.equal(batch.sequences[1], torch.tensor([11, 12, 0]))


if __name__ == "__main__":
    pytest.main([__file__])