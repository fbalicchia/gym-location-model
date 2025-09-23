# POI Prediction with Cross-Attention Transformer

A PyTorch implementation of a Transformer model with cross-attention mechanisms for next Point-of-Interest (POI) prediction. This project implements a sophisticated architecture that combines multi-modal embeddings, user context, geographical attention, and multiple prediction heads for comprehensive POI recommendation.

## 🚀 Key Features

### High-Level Architecture

The `POITransformerWithCrossAttention` class combines:
- **Multi-modal embeddings** (POI, category, time, location, region, etc.)
- **User context encoding** with preference modeling
- **Geographic attention** for spatial relationships
- **Transformer blocks** (some with cross-attention, some without)
- **Multiple output heads** for:
  - Next POI ranking
  - Next category prediction  
  - Time-to-next POI regression

### Cross-Attention Mechanisms

- **User Cross-Attention**: Attends to user preferences and behavioral patterns
- **Geographical Cross-Attention**: Models spatial relationships between POIs
- **Multi-Scale Attention**: Combines different types of contextual information
- **Alternating Architecture**: Efficient computation with cross-attention every N layers

### Advanced Features

- **Synthetic Data Generation**: Realistic POI trajectories with temporal and spatial patterns
- **Multi-Task Learning**: Joint optimization of ranking, classification, and regression
- **Comprehensive Metrics**: Hit rates, MRR, NDCG, accuracy, MSE, and more
- **Attention Analysis**: Interpretable attention pattern visualization
- **Flexible Configuration**: Easy customization of model architecture and training

## 📦 Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management.

```bash
# Clone the repository
git clone <repository-url>
cd poi-prediction-transformer

# Install dependencies with uv
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

### Dependencies

- PyTorch >= 2.0.0
- NumPy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- matplotlib >= 3.5.0
- tqdm >= 4.62.0
- pydantic >= 2.0.0
- wandb >= 0.15.0 (optional, for experiment tracking)

## 🎯 Quick Start

### Demo

Run the interactive demo to see the model in action:

```bash
python demo.py
```

This will:
- Create synthetic POI data with realistic patterns
- Initialize the cross-attention transformer model
- Show model architecture and sample predictions
- Optionally run a quick training session

### Training

Train the model with default settings:

```bash
poi-train
```

Or with custom parameters:

```bash
poi-train --d_model 256 --nhead 8 --num_layers 6 --batch_size 64 --num_epochs 100 --use_wandb
```

### Prediction

Make predictions with a trained model:

```bash
poi-predict --checkpoint ./checkpoints/best_model.pt --num_samples 1000 --top_k 10
```

## 🏗️ Architecture Details


#### 1. Multi-Modal Embedding Layer
```python
class MultiModalEmbedding(nn.Module):
    # Combines:
    # - POI ID embeddings
    # - Category embeddings  
    # - Temporal embeddings (hour, day, season)
    # - Spatial embeddings (coordinates, regions)
    # - Duration embeddings
```

#### 2. Cross-Attention Mechanisms
```python
class MultiScaleCrossAttention(nn.Module):
    # Implements:
    # - User context cross-attention
    # - Geographical cross-attention
    # - Temporal cross-attention
    # - Context fusion network
```

#### 3. Transformer Architecture
```python
class POITransformerWithCrossAttention(nn.Module):
    # Features:
    # - Alternating self-attention and cross-attention layers
    # - Causal or bidirectional attention masks
    # - Multiple prediction heads
    # - Attention pattern analysis
```

### Data Flow

1. **Input Processing**: Multi-modal features → Embeddings → Positional encoding
2. **Context Encoding**: User preferences → Context vectors
3. **Transformer Processing**: Self-attention ↔ Cross-attention layers
4. **Output Generation**: 
   - POI candidate scoring
   - Category classification
   - Duration regression

## 📊 Dataset

The project includes a sophisticated synthetic data generator that creates realistic POI trajectories:

### POI Database
- 26 different POI types (restaurants, offices, gyms, etc.)
- NYC-based geographical coordinates
- Category hierarchies and popularity scores
- Regional organization

### User Profiles
- Demographic-based preferences
- Activity level and location diversity
- Temporal regularity patterns
- Behavioral modeling

### Trajectory Generation
- **Temporal Patterns**: Morning routines, weekly shopping, monthly services
- **Spatial Patterns**: Distance-based preferences, regional constraints
- **Long-Range Dependencies**: Coffee → Office, Dinner → Cinema
- **User Personalization**: Individual preference modeling

## 🔧 Configuration

### Model Configuration
```python
@dataclass
class ModelConfig:
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    use_cross_attention: bool = True
    cross_attention_every_n_layers: int = 2
    max_distance: float = 50.0  # km
    # ... more options
```

### Training Configuration
```python
@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    patience: int = 10
    use_wandb: bool = False
    # ... more options
```

## 📈 Metrics and Evaluation

### Ranking Metrics
- **Hit Rate@K**: Percentage of correct POIs in top-K predictions
- **MRR@K**: Mean Reciprocal Rank for top-K predictions
- **NDCG@K**: Normalized Discounted Cumulative Gain

### Classification Metrics
- **Accuracy**: Category prediction accuracy
- **Precision/Recall/F1**: Macro-averaged metrics
- **Top-K Accuracy**: Multi-class top-K performance

### Regression Metrics
- **MSE/RMSE**: Duration prediction error
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

### Attention Analysis
- **Attention Entropy**: Measure of attention dispersion
- **Max Attention Weight**: Attention concentration
- **Cross-Attention Patterns**: User and geographical attention

## 🔬 Advanced Usage

### Custom Data

To use your own POI data:

```python
from poi_prediction.data.dataset import POIDataset
from poi_prediction.data.data_structures import TrajectorySequence, Visit

# Create your trajectory sequences
sequences = [TrajectorySequence(...), ...]

# Create dataset
dataset = POIDataset(sequences, generator, config, model_config)
```

### Model Customization

Extend the base model for specific use cases:

```python
class CustomPOITransformer(POITransformerWithCrossAttention):
    def __init__(self, config):
        super().__init__(config)
        # Add custom components
        self.custom_head = nn.Linear(config.d_model, custom_output_dim)
    
    def forward(self, batch_data):
        outputs = super().forward(batch_data)
        # Add custom predictions
        outputs.custom_predictions = self.custom_head(outputs.hidden_states)
        return outputs
```

### Attention Visualization

Analyze attention patterns:

```python
# Get attention weights
outputs = model(batch_data, return_attention=True)
attention_weights = outputs.attention_weights

# Analyze patterns
for layer_idx, layer_attn in enumerate(attention_weights):
    self_attn = layer_attn['self_attention']
    user_attn = layer_attn.get('user_attention')
    geo_attn = layer_attn.get('geo_attention')
    
    # Visualize or analyze attention patterns
```

## 🧪 Experiments and Results

### Baseline Comparisons

The model can be compared against:
- **Simple RNN/LSTM**: Sequential models without attention
- **Standard Transformer**: Self-attention only, no cross-attention
- **Matrix Factorization**: Traditional collaborative filtering
- **Markov Chain**: Simple transition-based models

### Ablation Studies

Test different components:
```bash
# No cross-attention
poi-train --use_cross_attention False

# Different attention frequencies
poi-train --cross_attention_every_n_layers 3

# Vary model dimensions
poi-train --d_model 256 --nhead 16
```


## 🔗 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Cross-attention inspiration
- [Location-Based Social Networks](https://www.amazon.com/Interest-Recommendation-Location-Based-Networks-SpringerBriefs/dp/9811313482) - POI recommendation background



**Note**: This is a research implementation focused on demonstrating cross-attention mechanisms in POI prediction. For production use, consider additional optimizations for scalability and efficiency.
