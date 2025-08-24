# ProteusNet Architecture Design Document

## Overview

ProteusNet is a novel transformer-based architecture that incorporates protein physics constraints and hierarchical processing for comprehensive protein understanding. The design builds upon standard transformer architectures while introducing three key innovations: physics-informed attention mechanisms, hierarchical multi-scale encoders, and evolutionary-aware representations.

The architecture processes protein sequences through multiple scales simultaneously, from individual amino acids to protein domains, while enforcing physical constraints through specialized attention mechanisms. This approach enables the model to capture both local structural patterns and global protein properties, leading to improved performance on function prediction, structure prediction, and stability analysis tasks.

## Architecture

### Core Components

The ProteusNet architecture consists of five main components:

1. **Embedding Layer**: Combines amino acid embeddings, positional encodings, and physics feature projections
2. **Hierarchical Encoder**: Multi-scale processing at amino acid, motif, and domain levels
3. **Physics-Informed Attention**: Attention mechanism incorporating distance bias and interaction potentials
4. **Task-Specific Heads**: Separate output layers for function, structure, and stability prediction
5. **Physics Consistency Module**: Enforces physical constraints during training

### Information Flow

```
Input Sequence → Embedding Layer → Hierarchical Encoder → Physics-Informed Attention → Task Heads → Outputs
                      ↑                    ↑                        ↑
                Physics Features    Cross-Level Fusion    Physics Consistency Loss
```

The architecture processes information through three hierarchical levels:
- **Level 1**: Amino acid-level features (full resolution)
- **Level 2**: Motif-level features (2x downsampled)  
- **Level 3**: Domain-level features (4x downsampled)

Cross-level attention mechanisms enable information exchange between scales, while physics-informed attention incorporates biological constraints at each level.

## Components and Interfaces

### ProteusNet Main Class

```python
class ProteusNet(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, max_seq_length, 
                 hierarchical_levels, physics_features):
    
    def forward(self, sequence, physics_features=None, evolutionary_features=None) -> Dict[str, torch.Tensor]:
```

**Inputs:**
- `sequence`: Tokenized amino acid sequence [batch_size, seq_length]
- `physics_features`: Optional physics constraints [batch_size, physics_dim]
- `evolutionary_features`: Optional evolutionary information [batch_size, evo_dim]

**Outputs:**
- `function_logits`: GO term predictions [batch_size, 1000]
- `structure_logits`: Secondary structure predictions [batch_size, seq_length, 8]
- `stability_score`: Thermostability prediction [batch_size, 1]
- `representations`: Learned protein representations [batch_size, seq_length, d_model]

### Physics-Informed Attention

```python
class PhysicsInformedAttention(nn.Module):
    def __init__(self, d_model, n_heads):
    
    def forward(self, x, physics_features=None, mask=None) -> torch.Tensor:
    
    def _compute_physics_bias(self, physics_features, seq_len) -> torch.Tensor:
```

**Key Features:**
- Distance-aware attention bias using learnable decay parameters
- Amino acid interaction matrix (25x25) for pairwise interaction potentials
- Secondary structure bias integration
- Graceful degradation when physics features are unavailable

**Physics Bias Computation:**
- Extracts distance matrices from physics features
- Applies learnable distance decay: `bias = -distance^decay_param * scale_param`
- Broadcasts bias across attention heads
- Combines with standard attention scores

### Hierarchical Encoder

```python
class HierarchicalEncoder(nn.Module):
    def __init__(self, d_model, n_heads, n_layers, levels=3):
    
    def forward(self, x) -> torch.Tensor:
```

**Multi-Scale Processing:**
- **Level 1**: Full resolution transformer encoder (amino acid level)
- **Level 2**: 2x downsampled features (motif level) 
- **Level 3**: 4x downsampled features (domain level)

**Cross-Level Integration:**
- Cross-attention between adjacent levels
- Feature upsampling using linear interpolation
- Concatenation and fusion of all levels
- Final projection to original dimensionality

### Training Components

```python
class ProteusNetTrainer:
    def train_epoch(self, dataloader) -> Dict[str, float]:
    def _compute_losses(self, outputs, targets) -> Dict[str, torch.Tensor]:

class PhysicsConsistencyLoss(nn.Module):
    def forward(self, representations, physics_features) -> torch.Tensor:
```

**Loss Functions:**
- Function prediction: Binary cross-entropy with logits
- Structure prediction: Cross-entropy for 8-class secondary structure
- Stability prediction: Mean squared error
- Physics consistency: Custom loss enforcing distance relationships

## Data Models

### Input Data Structure

```python
@dataclass
class ProteinBatch:
    sequences: torch.Tensor          # [batch_size, seq_length]
    function_labels: torch.Tensor    # [batch_size, 1000] - GO terms
    structure_labels: torch.Tensor   # [batch_size, seq_length] - SS labels
    stability_labels: torch.Tensor   # [batch_size, 1] - stability scores
    physics_features: Optional[torch.Tensor]  # [batch_size, physics_dim]
    evolutionary_features: Optional[torch.Tensor]  # [batch_size, evo_dim]
    masks: torch.Tensor             # [batch_size, seq_length]
```

### Configuration Schema

```yaml
model:
  vocab_size: int                   # Amino acid vocabulary size
  d_model: int                     # Model dimension
  n_heads: int                     # Number of attention heads
  n_layers: int                    # Total transformer layers
  max_seq_length: int              # Maximum sequence length
  hierarchical_levels: int         # Number of hierarchical levels
  physics_features: int            # Physics feature dimension

training:
  batch_size: int
  learning_rate: float
  weight_decay: float
  max_epochs: int
  loss_weights:                    # Weights for multi-task learning
    function_prediction: float
    structure_prediction: float
    stability_prediction: float
    physics_consistency: float

physics:
  enable_distance_bias: bool
  enable_interaction_matrix: bool
  enable_secondary_structure_bias: bool
  physics_regularization: float
```

### Physics Feature Encoding

Physics features are encoded as fixed-size vectors containing:
- **Distance Matrix**: Flattened upper triangular matrix of Cα distances
- **Interaction Energies**: Pairwise amino acid interaction potentials
- **Secondary Structure**: One-hot encoded 8-state secondary structure
- **Solvent Accessibility**: Relative solvent accessible surface area
- **Dihedral Angles**: Phi/psi backbone angles

## Error Handling

### Input Validation

- **Sequence Length**: Truncate or pad sequences to max_seq_length
- **Invalid Tokens**: Replace unknown amino acids with special UNK token
- **Missing Features**: Use zero tensors for optional physics/evolutionary features
- **Batch Size Mismatch**: Ensure all tensors have consistent batch dimensions

### Training Robustness

- **Gradient Clipping**: Prevent exploding gradients with configurable threshold
- **Loss Scaling**: Handle numerical instability in mixed precision training
- **NaN Detection**: Monitor for NaN values in loss computation and gradients
- **Memory Management**: Implement gradient checkpointing for large models

### Inference Handling

- **Device Compatibility**: Automatic device detection and tensor placement
- **Batch Processing**: Support variable batch sizes during evaluation
- **Missing Modalities**: Graceful handling when physics features unavailable
- **Output Validation**: Ensure output tensors have expected shapes and ranges

## Testing Strategy

### Unit Testing

**Component Tests:**
- `test_physics_attention.py`: Verify attention bias computation and physics integration
- `test_hierarchical_encoder.py`: Test multi-scale processing and cross-level attention
- `test_proteus_net.py`: Validate end-to-end forward pass and output shapes
- `test_loss_functions.py`: Verify custom loss computations and gradients

**Test Coverage:**
- Physics bias computation with various input shapes
- Hierarchical encoding with different sequence lengths
- Loss function behavior with missing modalities
- Configuration loading and validation

### Integration Testing

**Training Pipeline:**
- `test_training_loop.py`: Verify complete training iteration
- `test_multi_task_learning.py`: Test simultaneous optimization of multiple objectives
- `test_physics_consistency.py`: Validate physics constraint enforcement

**Data Pipeline:**
- `test_data_loading.py`: Verify batch construction and preprocessing
- `test_tokenization.py`: Test amino acid sequence tokenization
- `test_feature_extraction.py`: Validate physics feature computation

### Performance Testing

**Benchmarking:**
- Function prediction on GO dataset (target AUC > 0.85)
- Structure prediction on CASP dataset (target Q8 > 82%)
- Stability prediction on thermostability dataset (target correlation > 0.75)

**Ablation Studies:**
- Physics-informed attention vs standard attention
- Hierarchical processing vs single-scale processing
- Individual loss component contributions
- Different physics feature combinations

### Validation Strategy

**Cross-Validation:**
- 5-fold cross-validation on training datasets
- Temporal split validation for time-series protein data
- Phylogenetic split validation to test evolutionary generalization

**Baseline Comparisons:**
- Standard transformer (BERT-like) architecture
- Existing protein language models (ESM, ProtTrans)
- Traditional machine learning methods (SVM, Random Forest)

**Statistical Significance:**
- Bootstrap confidence intervals for performance metrics
- Paired t-tests for model comparisons
- Multiple hypothesis correction for ablation studies

This design provides a comprehensive framework for implementing ProteusNet while ensuring robustness, testability, and scientific rigor in the evaluation process.