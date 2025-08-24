"""
Physics-aware protein sequence tokenization.
"""

import re
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import torch


class ProteinTokenizer:
    """
    Physics-aware protein sequence tokenizer with special tokens and preprocessing.
    
    This tokenizer handles amino acid sequences with support for:
    - Standard 20 amino acids
    - Special tokens (PAD, UNK, CLS, SEP, MASK)
    - Ambiguous amino acids (B, Z, X)
    - Sequence length handling with truncation and padding
    """
    
    # Standard 20 amino acids
    STANDARD_AA = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                   'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    # Ambiguous amino acids
    AMBIGUOUS_AA = ['B', 'Z', 'X']  # B=N/D, Z=Q/E, X=any
    
    # Special tokens
    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<CLS>', '<SEP>', '<MASK>']
    
    def __init__(self, 
                 max_length: int = 1024,
                 add_special_tokens: bool = True,
                 handle_ambiguous: str = 'keep'):  # 'keep', 'remove', 'replace'
        """
        Initialize protein tokenizer.
        
        Args:
            max_length: Maximum sequence length
            add_special_tokens: Whether to add special tokens to vocabulary
            handle_ambiguous: How to handle ambiguous amino acids
                - 'keep': Keep ambiguous amino acids in vocabulary
                - 'remove': Remove sequences with ambiguous amino acids
                - 'replace': Replace ambiguous amino acids with UNK
        """
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.handle_ambiguous = handle_ambiguous
        
        # Build vocabulary
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        # Create token mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        # Special token IDs
        self.pad_token_id = self.token_to_id.get('<PAD>', 0)
        self.unk_token_id = self.token_to_id.get('<UNK>', 1)
        self.cls_token_id = self.token_to_id.get('<CLS>', 2)
        self.sep_token_id = self.token_to_id.get('<SEP>', 3)
        self.mask_token_id = self.token_to_id.get('<MASK>', 4)
    
    def _build_vocabulary(self) -> List[str]:
        """Build the tokenizer vocabulary."""
        vocab = []
        
        # Add special tokens first
        if self.add_special_tokens:
            vocab.extend(self.SPECIAL_TOKENS)
        
        # Add standard amino acids
        vocab.extend(self.STANDARD_AA)
        
        # Add ambiguous amino acids if keeping them
        if self.handle_ambiguous == 'keep':
            vocab.extend(self.AMBIGUOUS_AA)
        
        return vocab
    
    def tokenize(self, sequence: str) -> List[str]:
        """
        Tokenize a protein sequence into amino acid tokens.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            List of amino acid tokens
        """
        # Clean sequence - remove whitespace and convert to uppercase
        sequence = re.sub(r'\s+', '', sequence.upper())
        
        # Handle ambiguous amino acids
        if self.handle_ambiguous == 'remove':
            if any(aa in sequence for aa in self.AMBIGUOUS_AA):
                raise ValueError(f"Sequence contains ambiguous amino acids: {sequence}")
        
        # Tokenize into individual amino acids
        tokens = []
        for char in sequence:
            if char in self.token_to_id:
                tokens.append(char)
            elif self.handle_ambiguous == 'replace' and char in self.AMBIGUOUS_AA:
                tokens.append('<UNK>')
            else:
                tokens.append('<UNK>')
        
        return tokens
    
    def encode(self, 
               sequence: str, 
               add_special_tokens: bool = True,
               max_length: Optional[int] = None,
               padding: bool = True,
               truncation: bool = True,
               return_tensors: Optional[str] = None) -> Dict[str, Union[List[int], torch.Tensor]]:
        """
        Encode a protein sequence to token IDs.
        
        Args:
            sequence: Protein sequence string
            add_special_tokens: Whether to add CLS and SEP tokens
            max_length: Maximum sequence length (uses self.max_length if None)
            padding: Whether to pad to max_length
            truncation: Whether to truncate if longer than max_length
            return_tensors: Format of return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary with 'input_ids' and 'attention_mask'
        """
        if max_length is None:
            max_length = self.max_length
        
        # Tokenize sequence
        tokens = self.tokenize(sequence)
        
        # Add special tokens
        if add_special_tokens and self.add_special_tokens:
            tokens = ['<CLS>'] + tokens + ['<SEP>']
        
        # Convert to IDs
        input_ids = [self.token_to_id[token] for token in tokens]
        
        # Handle truncation
        if truncation and len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            # Ensure SEP token at the end if we had special tokens
            if add_special_tokens and self.add_special_tokens:
                input_ids[-1] = self.sep_token_id
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Handle padding
        if padding and len(input_ids) < max_length:
            pad_length = max_length - len(input_ids)
            input_ids.extend([self.pad_token_id] * pad_length)
            attention_mask.extend([0] * pad_length)
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        
        return result
    
    def encode_batch(self, 
                     sequences: List[str],
                     add_special_tokens: bool = True,
                     max_length: Optional[int] = None,
                     padding: bool = True,
                     truncation: bool = True,
                     return_tensors: Optional[str] = None) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """
        Encode a batch of protein sequences.
        
        Args:
            sequences: List of protein sequence strings
            add_special_tokens: Whether to add CLS and SEP tokens
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate if longer than max_length
            return_tensors: Format of return tensors ('pt' for PyTorch)
            
        Returns:
            Dictionary with batched 'input_ids' and 'attention_mask'
        """
        batch_input_ids = []
        batch_attention_mask = []
        
        for sequence in sequences:
            encoded = self.encode(
                sequence=sequence,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                padding=padding,
                truncation=truncation,
                return_tensors=None  # Handle tensor conversion at batch level
            )
            batch_input_ids.append(encoded['input_ids'])
            batch_attention_mask.append(encoded['attention_mask'])
        
        result = {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_mask
        }
        
        # Convert to tensors if requested
        if return_tensors == 'pt':
            result = {k: torch.tensor(v, dtype=torch.long) for k, v in result.items()}
        
        return result
    
    def decode(self, token_ids: Union[List[int], torch.Tensor], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to protein sequence.
        
        Args:
            token_ids: List of token IDs or tensor
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded protein sequence string
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if skip_special_tokens and token in self.SPECIAL_TOKENS:
                    continue
                tokens.append(token)
        
        return ''.join(tokens)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get the tokenizer vocabulary."""
        return self.token_to_id.copy()
    
    def save_vocabulary(self, vocab_path: str) -> None:
        """Save vocabulary to file."""
        import json
        with open(vocab_path, 'w') as f:
            json.dump(self.token_to_id, f, indent=2)
    
    def load_vocabulary(self, vocab_path: str) -> None:
        """Load vocabulary from file."""
        import json
        with open(vocab_path, 'r') as f:
            self.token_to_id = json.load(f)
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.vocab = list(self.token_to_id.keys())
        self.vocab_size = len(self.vocab)


class PhysicsFeatureExtractor:
    """
    Extract physics-based features from protein sequences.
    
    This class computes various physical and chemical properties that can be used
    as additional features for the physics-informed attention mechanism.
    """
    
    # Amino acid properties (normalized values)
    AA_PROPERTIES = {
        'A': {'hydrophobicity': 0.62, 'volume': 67.0, 'charge': 0.0, 'polarity': 0.0},
        'R': {'hydrophobicity': -2.53, 'volume': 148.0, 'charge': 1.0, 'polarity': 1.0},
        'N': {'hydrophobicity': -0.78, 'volume': 96.0, 'charge': 0.0, 'polarity': 1.0},
        'D': {'hydrophobicity': -0.90, 'volume': 91.0, 'charge': -1.0, 'polarity': 1.0},
        'C': {'hydrophobicity': 0.29, 'volume': 86.0, 'charge': 0.0, 'polarity': 0.0},
        'Q': {'hydrophobicity': -0.85, 'volume': 114.0, 'charge': 0.0, 'polarity': 1.0},
        'E': {'hydrophobicity': -0.74, 'volume': 109.0, 'charge': -1.0, 'polarity': 1.0},
        'G': {'hydrophobicity': 0.48, 'volume': 48.0, 'charge': 0.0, 'polarity': 0.0},
        'H': {'hydrophobicity': -0.40, 'volume': 118.0, 'charge': 0.5, 'polarity': 1.0},
        'I': {'hydrophobicity': 1.38, 'volume': 124.0, 'charge': 0.0, 'polarity': 0.0},
        'L': {'hydrophobicity': 1.06, 'volume': 124.0, 'charge': 0.0, 'polarity': 0.0},
        'K': {'hydrophobicity': -1.50, 'volume': 135.0, 'charge': 1.0, 'polarity': 1.0},
        'M': {'hydrophobicity': 0.64, 'volume': 124.0, 'charge': 0.0, 'polarity': 0.0},
        'F': {'hydrophobicity': 1.19, 'volume': 135.0, 'charge': 0.0, 'polarity': 0.0},
        'P': {'hydrophobicity': 0.12, 'volume': 90.0, 'charge': 0.0, 'polarity': 0.0},
        'S': {'hydrophobicity': -0.18, 'volume': 73.0, 'charge': 0.0, 'polarity': 1.0},
        'T': {'hydrophobicity': -0.05, 'volume': 93.0, 'charge': 0.0, 'polarity': 1.0},
        'W': {'hydrophobicity': 0.81, 'volume': 163.0, 'charge': 0.0, 'polarity': 0.0},
        'Y': {'hydrophobicity': 0.26, 'volume': 141.0, 'charge': 0.0, 'polarity': 1.0},
        'V': {'hydrophobicity': 1.08, 'volume': 105.0, 'charge': 0.0, 'polarity': 0.0},
    }
    
    def __init__(self):
        """Initialize physics feature extractor."""
        # Normalize properties for better numerical stability
        self._normalize_properties()
    
    def _normalize_properties(self):
        """Normalize amino acid properties to [0, 1] range."""
        properties = ['hydrophobicity', 'volume', 'charge', 'polarity']
        
        for prop in properties:
            values = [self.AA_PROPERTIES[aa][prop] for aa in self.AA_PROPERTIES]
            min_val, max_val = min(values), max(values)
            
            if max_val > min_val:
                for aa in self.AA_PROPERTIES:
                    original = self.AA_PROPERTIES[aa][prop]
                    self.AA_PROPERTIES[aa][prop] = (original - min_val) / (max_val - min_val)
    
    def extract_sequence_features(self, sequence: str) -> np.ndarray:
        """
        Extract physics features for a protein sequence.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Feature vector of shape [feature_dim]
        """
        sequence = sequence.upper().replace(' ', '')
        
        # Initialize feature arrays
        hydrophobicity = []
        volume = []
        charge = []
        polarity = []
        
        for aa in sequence:
            if aa in self.AA_PROPERTIES:
                props = self.AA_PROPERTIES[aa]
                hydrophobicity.append(props['hydrophobicity'])
                volume.append(props['volume'])
                charge.append(props['charge'])
                polarity.append(props['polarity'])
            else:
                # Use average values for unknown amino acids
                hydrophobicity.append(0.5)
                volume.append(0.5)
                charge.append(0.0)
                polarity.append(0.5)
        
        # Compute sequence-level statistics
        features = []
        
        for prop_values in [hydrophobicity, volume, charge, polarity]:
            if prop_values:
                features.extend([
                    np.mean(prop_values),      # Mean
                    np.std(prop_values),       # Standard deviation
                    np.min(prop_values),       # Minimum
                    np.max(prop_values),       # Maximum
                ])
            else:
                features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Add sequence length (normalized)
        features.append(min(len(sequence) / 1000.0, 1.0))  # Normalize by typical max length
        
        # Add composition features (fraction of each property type)
        total_aa = len(sequence)
        if total_aa > 0:
            hydrophobic_count = sum(1 for aa in sequence if aa in self.AA_PROPERTIES and self.AA_PROPERTIES[aa]['hydrophobicity'] > 0.5)
            charged_count = sum(1 for aa in sequence if aa in self.AA_PROPERTIES and abs(self.AA_PROPERTIES[aa]['charge']) > 0.1)
            polar_count = sum(1 for aa in sequence if aa in self.AA_PROPERTIES and self.AA_PROPERTIES[aa]['polarity'] > 0.5)
            
            features.extend([
                hydrophobic_count / total_aa,
                charged_count / total_aa,
                polar_count / total_aa
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        return np.array(features, dtype=np.float32)
    
    def extract_pairwise_features(self, sequence: str, max_distance: int = 50) -> np.ndarray:
        """
        Extract pairwise interaction features (simplified distance matrix).
        
        Args:
            sequence: Protein sequence string
            max_distance: Maximum sequence distance to consider
            
        Returns:
            Flattened upper triangular matrix of pairwise features
        """
        sequence = sequence.upper().replace(' ', '')
        seq_len = len(sequence)
        
        # Create simplified interaction matrix based on amino acid properties
        interaction_matrix = np.zeros((seq_len, seq_len), dtype=np.float32)
        
        for i in range(seq_len):
            for j in range(i + 1, min(i + max_distance + 1, seq_len)):
                aa1, aa2 = sequence[i], sequence[j]
                
                if aa1 in self.AA_PROPERTIES and aa2 in self.AA_PROPERTIES:
                    props1 = self.AA_PROPERTIES[aa1]
                    props2 = self.AA_PROPERTIES[aa2]
                    
                    # Compute interaction strength based on properties
                    hydrophobic_interaction = props1['hydrophobicity'] * props2['hydrophobicity']
                    charge_interaction = props1['charge'] * props2['charge']
                    
                    # Distance decay
                    distance_factor = 1.0 / (1.0 + (j - i))
                    
                    interaction_strength = (hydrophobic_interaction + charge_interaction) * distance_factor
                    interaction_matrix[i, j] = interaction_strength
        
        # Return flattened upper triangular matrix
        return interaction_matrix[np.triu_indices(seq_len, k=1)]
    
    def get_feature_dim(self) -> int:
        """Get the dimension of sequence-level features."""
        # 4 properties * 4 statistics + 1 length + 3 composition = 20 features
        return 20