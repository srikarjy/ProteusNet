"""
Protein batch data structures and validation utilities.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import numpy as np


@dataclass
class ProteinBatch:
    """
    Batch data structure for protein sequences with all required tensor fields.
    
    This class holds batched protein data including sequences, labels, and optional
    physics/evolutionary features for multi-task learning.
    """
    
    # Core sequence data
    sequences: torch.Tensor              # [batch_size, seq_length] - tokenized amino acid sequences
    masks: torch.Tensor                  # [batch_size, seq_length] - attention masks
    
    # Task-specific labels
    function_labels: Optional[torch.Tensor] = None    # [batch_size, num_functions] - GO term labels
    structure_labels: Optional[torch.Tensor] = None  # [batch_size, seq_length] - secondary structure
    stability_labels: Optional[torch.Tensor] = None  # [batch_size, 1] - thermostability scores
    
    # Optional features
    physics_features: Optional[torch.Tensor] = None      # [batch_size, physics_dim] - distance matrices, etc.
    evolutionary_features: Optional[torch.Tensor] = None # [batch_size, evo_dim] - MSA features, etc.
    
    # Metadata
    sequence_lengths: Optional[torch.Tensor] = None      # [batch_size] - actual sequence lengths
    protein_ids: Optional[List[str]] = None              # List of protein identifiers
    
    def __post_init__(self):
        """Validate tensor shapes and data types after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate tensor shapes and data types.
        
        Raises:
            ValueError: If tensor shapes or types are invalid
        """
        batch_size, seq_length = self.sequences.shape
        
        # Validate core tensors
        if not isinstance(self.sequences, torch.Tensor):
            raise ValueError("sequences must be a torch.Tensor")
        if not isinstance(self.masks, torch.Tensor):
            raise ValueError("masks must be a torch.Tensor")
        
        if self.sequences.dtype not in [torch.long, torch.int32, torch.int64]:
            raise ValueError("sequences must have integer dtype")
        if self.masks.dtype not in [torch.bool, torch.float32, torch.long]:
            raise ValueError("masks must have bool or numeric dtype")
        
        # Validate shapes match
        if self.masks.shape != (batch_size, seq_length):
            raise ValueError(f"masks shape {self.masks.shape} doesn't match sequences shape {self.sequences.shape}")
        
        # Validate optional labels
        if self.function_labels is not None:
            if self.function_labels.shape[0] != batch_size:
                raise ValueError(f"function_labels batch size {self.function_labels.shape[0]} doesn't match {batch_size}")
            if self.function_labels.dtype not in [torch.float32, torch.float16, torch.long]:
                raise ValueError("function_labels must have float or long dtype")
        
        if self.structure_labels is not None:
            if self.structure_labels.shape != (batch_size, seq_length):
                raise ValueError(f"structure_labels shape {self.structure_labels.shape} doesn't match sequences shape")
            if self.structure_labels.dtype not in [torch.long, torch.int32, torch.int64]:
                raise ValueError("structure_labels must have integer dtype")
        
        if self.stability_labels is not None:
            if self.stability_labels.shape[0] != batch_size:
                raise ValueError(f"stability_labels batch size {self.stability_labels.shape[0]} doesn't match {batch_size}")
            if self.stability_labels.dtype not in [torch.float32, torch.float16]:
                raise ValueError("stability_labels must have float dtype")
        
        # Validate optional features
        if self.physics_features is not None:
            if self.physics_features.shape[0] != batch_size:
                raise ValueError(f"physics_features batch size {self.physics_features.shape[0]} doesn't match {batch_size}")
            if self.physics_features.dtype not in [torch.float32, torch.float16]:
                raise ValueError("physics_features must have float dtype")
        
        if self.evolutionary_features is not None:
            if self.evolutionary_features.shape[0] != batch_size:
                raise ValueError(f"evolutionary_features batch size {self.evolutionary_features.shape[0]} doesn't match {batch_size}")
            if self.evolutionary_features.dtype not in [torch.float32, torch.float16]:
                raise ValueError("evolutionary_features must have float dtype")
        
        # Validate sequence lengths
        if self.sequence_lengths is not None:
            if self.sequence_lengths.shape != (batch_size,):
                raise ValueError(f"sequence_lengths shape {self.sequence_lengths.shape} doesn't match batch size {batch_size}")
            if torch.any(self.sequence_lengths > seq_length):
                raise ValueError("sequence_lengths cannot exceed max sequence length")
            if torch.any(self.sequence_lengths <= 0):
                raise ValueError("sequence_lengths must be positive")
        
        # Validate protein IDs
        if self.protein_ids is not None:
            if len(self.protein_ids) != batch_size:
                raise ValueError(f"protein_ids length {len(self.protein_ids)} doesn't match batch size {batch_size}")
    
    def to(self, device: torch.device) -> 'ProteinBatch':
        """
        Move all tensors to the specified device.
        
        Args:
            device: Target device
            
        Returns:
            New ProteinBatch with tensors on target device
        """
        return ProteinBatch(
            sequences=self.sequences.to(device),
            masks=self.masks.to(device),
            function_labels=self.function_labels.to(device) if self.function_labels is not None else None,
            structure_labels=self.structure_labels.to(device) if self.structure_labels is not None else None,
            stability_labels=self.stability_labels.to(device) if self.stability_labels is not None else None,
            physics_features=self.physics_features.to(device) if self.physics_features is not None else None,
            evolutionary_features=self.evolutionary_features.to(device) if self.evolutionary_features is not None else None,
            sequence_lengths=self.sequence_lengths.to(device) if self.sequence_lengths is not None else None,
            protein_ids=self.protein_ids  # Keep on CPU
        )
    
    def pin_memory(self) -> 'ProteinBatch':
        """
        Pin memory for all tensors to enable faster GPU transfer.
        
        Returns:
            New ProteinBatch with pinned memory tensors
        """
        # Only pin memory if tensors are on CPU
        def pin_if_cpu(tensor):
            if tensor is not None and tensor.device.type == 'cpu':
                return tensor.pin_memory()
            return tensor
        
        return ProteinBatch(
            sequences=pin_if_cpu(self.sequences),
            masks=pin_if_cpu(self.masks),
            function_labels=pin_if_cpu(self.function_labels),
            structure_labels=pin_if_cpu(self.structure_labels),
            stability_labels=pin_if_cpu(self.stability_labels),
            physics_features=pin_if_cpu(self.physics_features),
            evolutionary_features=pin_if_cpu(self.evolutionary_features),
            sequence_lengths=pin_if_cpu(self.sequence_lengths),
            protein_ids=self.protein_ids
        )
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.sequences.shape[0]
    
    @property
    def seq_length(self) -> int:
        """Get sequence length."""
        return self.sequences.shape[1]


class BatchConstructor:
    """Utility class for constructing and padding protein batches."""
    
    def __init__(self, pad_token_id: int = 0, max_length: int = 1024):
        """
        Initialize batch constructor.
        
        Args:
            pad_token_id: Token ID used for padding
            max_length: Maximum sequence length
        """
        self.pad_token_id = pad_token_id
        self.max_length = max_length
    
    def collate_fn(self, samples: List[Dict[str, Any]]) -> ProteinBatch:
        """
        Collate function for DataLoader to create batches.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            ProteinBatch object
        """
        batch_size = len(samples)
        
        # Extract sequences and determine max length
        sequences = [sample['sequence'] for sample in samples]
        seq_lengths = [len(seq) for seq in sequences]
        max_len = min(max(seq_lengths), self.max_length)
        
        # Clip sequence lengths to max_len for validation
        clipped_seq_lengths = [min(length, max_len) for length in seq_lengths]
        
        # Pad sequences
        padded_sequences = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        masks = torch.zeros((batch_size, max_len), dtype=torch.bool)
        
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_len)
            padded_sequences[i, :length] = torch.tensor(seq[:length], dtype=torch.long)
            masks[i, :length] = True
        
        # Extract labels if available
        function_labels = None
        if 'function_labels' in samples[0] and samples[0]['function_labels'] is not None:
            function_labels = torch.stack([
                torch.tensor(sample['function_labels'], dtype=torch.float32) 
                for sample in samples
            ])
        
        structure_labels = None
        if 'structure_labels' in samples[0] and samples[0]['structure_labels'] is not None:
            structure_labels = torch.full((batch_size, max_len), -1, dtype=torch.long)
            for i, sample in enumerate(samples):
                if sample['structure_labels'] is not None:
                    labels = sample['structure_labels']
                    length = min(len(labels), max_len)
                    structure_labels[i, :length] = torch.tensor(labels[:length], dtype=torch.long)
        
        stability_labels = None
        if 'stability_labels' in samples[0] and samples[0]['stability_labels'] is not None:
            stability_labels = torch.tensor([
                sample['stability_labels'] for sample in samples
            ], dtype=torch.float32).unsqueeze(1)
        
        # Extract features if available
        physics_features = None
        if 'physics_features' in samples[0] and samples[0]['physics_features'] is not None:
            physics_features = torch.stack([
                torch.tensor(sample['physics_features'], dtype=torch.float32)
                for sample in samples
            ])
        
        evolutionary_features = None
        if 'evolutionary_features' in samples[0] and samples[0]['evolutionary_features'] is not None:
            evolutionary_features = torch.stack([
                torch.tensor(sample['evolutionary_features'], dtype=torch.float32)
                for sample in samples
            ])
        
        # Extract metadata
        sequence_lengths = torch.tensor(clipped_seq_lengths, dtype=torch.long)
        protein_ids = [sample.get('protein_id', f'protein_{i}') for i, sample in enumerate(samples)]
        
        return ProteinBatch(
            sequences=padded_sequences,
            masks=masks,
            function_labels=function_labels,
            structure_labels=structure_labels,
            stability_labels=stability_labels,
            physics_features=physics_features,
            evolutionary_features=evolutionary_features,
            sequence_lengths=sequence_lengths,
            protein_ids=protein_ids
        )
    
    def pad_sequence(self, sequence: List[int], target_length: int) -> Tuple[List[int], List[bool]]:
        """
        Pad a single sequence to target length.
        
        Args:
            sequence: Input sequence as list of token IDs
            target_length: Target length for padding
            
        Returns:
            Tuple of (padded_sequence, attention_mask)
        """
        if len(sequence) >= target_length:
            # Truncate if too long
            padded_seq = sequence[:target_length]
            mask = [True] * target_length
        else:
            # Pad if too short
            padded_seq = sequence + [self.pad_token_id] * (target_length - len(sequence))
            mask = [True] * len(sequence) + [False] * (target_length - len(sequence))
        
        return padded_seq, mask
    
    def create_batch_from_sequences(self, sequences: List[List[int]], **kwargs) -> ProteinBatch:
        """
        Create a batch from a list of tokenized sequences.
        
        Args:
            sequences: List of tokenized sequences
            **kwargs: Additional data (labels, features, etc.)
            
        Returns:
            ProteinBatch object
        """
        samples = []
        for i, seq in enumerate(sequences):
            sample = {'sequence': seq}
            
            # Add any additional data
            for key, value in kwargs.items():
                if isinstance(value, (list, tuple)) and len(value) == len(sequences):
                    sample[key] = value[i]
                else:
                    sample[key] = value
            
            samples.append(sample)
        
        return self.collate_fn(samples)