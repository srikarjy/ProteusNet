"""
Unit tests for protein tokenizer and preprocessing.
"""

import pytest
import torch
import numpy as np
from src.data.protein_tokenizer import ProteinTokenizer, PhysicsFeatureExtractor


class TestProteinTokenizer:
    """Test cases for ProteinTokenizer class."""
    
    def test_initialization_default(self):
        """Test default tokenizer initialization."""
        tokenizer = ProteinTokenizer()
        
        assert tokenizer.max_length == 1024
        assert tokenizer.add_special_tokens == True
        assert tokenizer.handle_ambiguous == 'keep'
        assert tokenizer.vocab_size == 28  # 5 special + 20 standard + 3 ambiguous
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 1
    
    def test_initialization_custom(self):
        """Test custom tokenizer initialization."""
        tokenizer = ProteinTokenizer(
            max_length=512,
            add_special_tokens=False,
            handle_ambiguous='remove'
        )
        
        assert tokenizer.max_length == 512
        assert tokenizer.add_special_tokens == False
        assert tokenizer.handle_ambiguous == 'remove'
        assert tokenizer.vocab_size == 20  # Only standard amino acids
    
    def test_vocabulary_building(self):
        """Test vocabulary building with different settings."""
        # With special tokens and ambiguous
        tokenizer1 = ProteinTokenizer(handle_ambiguous='keep')
        expected_vocab1 = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>'] + ProteinTokenizer.STANDARD_AA + ProteinTokenizer.AMBIGUOUS_AA
        assert tokenizer1.vocab == expected_vocab1
        
        # Without special tokens
        tokenizer2 = ProteinTokenizer(add_special_tokens=False, handle_ambiguous='keep')
        expected_vocab2 = ProteinTokenizer.STANDARD_AA + ProteinTokenizer.AMBIGUOUS_AA
        assert tokenizer2.vocab == expected_vocab2
        
        # Without ambiguous amino acids
        tokenizer3 = ProteinTokenizer(handle_ambiguous='remove')
        expected_vocab3 = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>'] + ProteinTokenizer.STANDARD_AA
        assert tokenizer3.vocab == expected_vocab3
    
    def test_tokenize_basic(self):
        """Test basic sequence tokenization."""
        tokenizer = ProteinTokenizer()
        
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == list(sequence)
        assert len(tokens) == 20
    
    def test_tokenize_with_spaces(self):
        """Test tokenization with whitespace removal."""
        tokenizer = ProteinTokenizer()
        
        sequence = "AC DE FG HI"
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    
    def test_tokenize_lowercase(self):
        """Test tokenization with lowercase conversion."""
        tokenizer = ProteinTokenizer()
        
        sequence = "acdefg"
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == ['A', 'C', 'D', 'E', 'F', 'G']
    
    def test_tokenize_unknown_amino_acids(self):
        """Test tokenization with unknown amino acids."""
        tokenizer = ProteinTokenizer()
        
        sequence = "ACJOU"  # J, O, U are not standard
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == ['A', 'C', '<UNK>', '<UNK>', '<UNK>']
    
    def test_tokenize_ambiguous_keep(self):
        """Test tokenization keeping ambiguous amino acids."""
        tokenizer = ProteinTokenizer(handle_ambiguous='keep')
        
        sequence = "ACBZX"
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == ['A', 'C', 'B', 'Z', 'X']
    
    def test_tokenize_ambiguous_replace(self):
        """Test tokenization replacing ambiguous amino acids."""
        tokenizer = ProteinTokenizer(handle_ambiguous='replace')
        
        sequence = "ACBZX"
        tokens = tokenizer.tokenize(sequence)
        
        assert tokens == ['A', 'C', '<UNK>', '<UNK>', '<UNK>']
    
    def test_tokenize_ambiguous_remove(self):
        """Test tokenization removing sequences with ambiguous amino acids."""
        tokenizer = ProteinTokenizer(handle_ambiguous='remove')
        
        sequence = "ACBZX"
        with pytest.raises(ValueError, match="Sequence contains ambiguous amino acids"):
            tokenizer.tokenize(sequence)
    
    def test_encode_basic(self):
        """Test basic sequence encoding."""
        tokenizer = ProteinTokenizer()
        
        sequence = "ACG"
        result = tokenizer.encode(sequence)
        
        # Should be: <CLS> A C G <SEP>
        expected_ids = [2, 5, 9, 12, 3]  # CLS, A, C, G, SEP
        assert result['input_ids'][:5] == expected_ids
        assert result['attention_mask'][:5] == [1, 1, 1, 1, 1]
    
    def test_encode_no_special_tokens(self):
        """Test encoding without special tokens."""
        tokenizer = ProteinTokenizer()
        
        sequence = "ACG"
        result = tokenizer.encode(sequence, add_special_tokens=False)
        
        # Should be: A C G
        expected_ids = [5, 9, 12]  # A, C, G
        assert result['input_ids'][:3] == expected_ids
        assert result['attention_mask'][:3] == [1, 1, 1]
    
    def test_encode_with_padding(self):
        """Test encoding with padding."""
        tokenizer = ProteinTokenizer(max_length=10)
        
        sequence = "ACG"
        result = tokenizer.encode(sequence, padding=True)
        
        assert len(result['input_ids']) == 10
        assert len(result['attention_mask']) == 10
        assert result['input_ids'][5:] == [0] * 5  # Padding tokens
        assert result['attention_mask'][5:] == [0] * 5  # Padding mask
    
    def test_encode_with_truncation(self):
        """Test encoding with truncation."""
        tokenizer = ProteinTokenizer(max_length=5)
        
        sequence = "ACDEFGHIKLMN"  # Long sequence
        result = tokenizer.encode(sequence, truncation=True)
        
        assert len(result['input_ids']) == 5
        assert result['input_ids'][-1] == tokenizer.sep_token_id  # Should end with SEP
    
    def test_encode_return_tensors(self):
        """Test encoding with tensor return."""
        tokenizer = ProteinTokenizer()
        
        sequence = "ACG"
        result = tokenizer.encode(sequence, return_tensors='pt')
        
        assert isinstance(result['input_ids'], torch.Tensor)
        assert isinstance(result['attention_mask'], torch.Tensor)
        assert result['input_ids'].dtype == torch.long
    
    def test_encode_batch(self):
        """Test batch encoding."""
        tokenizer = ProteinTokenizer(max_length=10)
        
        sequences = ["ACG", "DEFGH", "I"]
        result = tokenizer.encode_batch(sequences, return_tensors='pt')
        
        assert result['input_ids'].shape == (3, 10)
        assert result['attention_mask'].shape == (3, 10)
        assert isinstance(result['input_ids'], torch.Tensor)
    
    def test_decode_basic(self):
        """Test basic sequence decoding."""
        tokenizer = ProteinTokenizer()
        
        # Encode then decode
        sequence = "ACDEFG"
        encoded = tokenizer.encode(sequence, return_tensors='pt')
        decoded = tokenizer.decode(encoded['input_ids'])
        
        assert decoded == sequence
    
    def test_decode_with_special_tokens(self):
        """Test decoding with special tokens."""
        tokenizer = ProteinTokenizer()
        
        token_ids = [2, 5, 9, 12, 3, 0, 0]  # CLS A C G SEP PAD PAD
        decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
        
        assert decoded == "ACG"
    
    def test_decode_keep_special_tokens(self):
        """Test decoding keeping special tokens."""
        tokenizer = ProteinTokenizer()
        
        token_ids = [2, 5, 9, 12, 3]  # CLS A C G SEP
        decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        
        assert decoded == "<CLS>ACG<SEP>"
    
    def test_get_vocab(self):
        """Test vocabulary retrieval."""
        tokenizer = ProteinTokenizer()
        vocab = tokenizer.get_vocab()
        
        assert isinstance(vocab, dict)
        assert len(vocab) == tokenizer.vocab_size
        assert vocab['<PAD>'] == 0
        assert vocab['A'] == 5  # First amino acid after special tokens


class TestPhysicsFeatureExtractor:
    """Test cases for PhysicsFeatureExtractor class."""
    
    def test_initialization(self):
        """Test feature extractor initialization."""
        extractor = PhysicsFeatureExtractor()
        
        # Check that properties are normalized
        for aa in extractor.AA_PROPERTIES:
            props = extractor.AA_PROPERTIES[aa]
            assert 0.0 <= props['hydrophobicity'] <= 1.0
            assert 0.0 <= props['volume'] <= 1.0
            assert props['polarity'] in [0.0, 1.0]  # Binary property
    
    def test_extract_sequence_features_basic(self):
        """Test basic sequence feature extraction."""
        extractor = PhysicsFeatureExtractor()
        
        sequence = "ACDEFG"
        features = extractor.extract_sequence_features(sequence)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape == (extractor.get_feature_dim(),)
        assert len(features) == 20  # Expected feature dimension
    
    def test_extract_sequence_features_empty(self):
        """Test feature extraction for empty sequence."""
        extractor = PhysicsFeatureExtractor()
        
        sequence = ""
        features = extractor.extract_sequence_features(sequence)
        
        assert features.shape == (extractor.get_feature_dim(),)
        # Should handle empty sequence gracefully
        assert not np.any(np.isnan(features))
    
    def test_extract_sequence_features_unknown_aa(self):
        """Test feature extraction with unknown amino acids."""
        extractor = PhysicsFeatureExtractor()
        
        sequence = "ACJOU"  # J, O, U are unknown
        features = extractor.extract_sequence_features(sequence)
        
        assert features.shape == (extractor.get_feature_dim(),)
        assert not np.any(np.isnan(features))
    
    def test_extract_sequence_features_properties(self):
        """Test that extracted features capture expected properties."""
        extractor = PhysicsFeatureExtractor()
        
        # Test with hydrophobic sequence
        hydrophobic_seq = "AILMFWYV"
        hydrophobic_features = extractor.extract_sequence_features(hydrophobic_seq)
        
        # Test with hydrophilic sequence
        hydrophilic_seq = "RNDEQKST"
        hydrophilic_features = extractor.extract_sequence_features(hydrophilic_seq)
        
        # Hydrophobic sequence should have higher hydrophobicity mean
        assert hydrophobic_features[0] > hydrophilic_features[0]  # Mean hydrophobicity
    
    def test_extract_pairwise_features(self):
        """Test pairwise feature extraction."""
        extractor = PhysicsFeatureExtractor()
        
        sequence = "ACDEFG"
        features = extractor.extract_pairwise_features(sequence, max_distance=10)
        
        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        
        # Should be upper triangular matrix flattened
        seq_len = len(sequence)
        expected_length = seq_len * (seq_len - 1) // 2
        assert len(features) <= expected_length  # May be less due to max_distance
    
    def test_extract_pairwise_features_max_distance(self):
        """Test pairwise features with distance limit."""
        extractor = PhysicsFeatureExtractor()
        
        sequence = "ACDEFGHIKLMN"
        features_short = extractor.extract_pairwise_features(sequence, max_distance=2)
        features_long = extractor.extract_pairwise_features(sequence, max_distance=10)
        
        # Shorter distance should give fewer features
        assert len(features_short) <= len(features_long)
    
    def test_get_feature_dim(self):
        """Test feature dimension getter."""
        extractor = PhysicsFeatureExtractor()
        
        dim = extractor.get_feature_dim()
        assert dim == 20
        
        # Verify it matches actual feature extraction
        features = extractor.extract_sequence_features("ACDEFG")
        assert len(features) == dim


class TestTokenizerIntegration:
    """Integration tests for tokenizer with other components."""
    
    def test_tokenizer_with_batch_constructor(self):
        """Test tokenizer integration with batch constructor."""
        from src.data.protein_batch import BatchConstructor
        
        tokenizer = ProteinTokenizer(max_length=20)
        constructor = BatchConstructor(pad_token_id=tokenizer.pad_token_id, max_length=20)
        
        sequences = ["ACDEFG", "HIKLMN", "PQRST"]
        
        # Tokenize sequences
        tokenized_sequences = []
        for seq in sequences:
            encoded = tokenizer.encode(seq, padding=False, add_special_tokens=False)
            tokenized_sequences.append(encoded['input_ids'])
        
        # Create batch
        batch = constructor.create_batch_from_sequences(tokenized_sequences)
        
        assert batch.batch_size == 3
        assert batch.seq_length <= 20
        
        # Verify we can decode back
        for i, original_seq in enumerate(sequences):
            decoded = tokenizer.decode(batch.sequences[i], skip_special_tokens=True)
            # Remove padding
            decoded = decoded.replace('<PAD>', '')
            assert decoded == original_seq
    
    def test_physics_features_integration(self):
        """Test physics feature extractor integration."""
        tokenizer = ProteinTokenizer()
        extractor = PhysicsFeatureExtractor()
        
        sequences = ["ACDEFG", "HIKLMN"]
        
        # Extract features for each sequence
        physics_features = []
        for seq in sequences:
            features = extractor.extract_sequence_features(seq)
            physics_features.append(features)
        
        # Should have consistent feature dimensions
        assert all(len(f) == extractor.get_feature_dim() for f in physics_features)
        assert len(physics_features) == len(sequences)


if __name__ == "__main__":
    pytest.main([__file__])