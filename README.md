# Gym-Location-Models

**Gym-Location-Models** is a personal research playground for experimenting with machine learning models in the **location intelligence domain** during my own time. This repository serves as a sandbox for implementing, testing, and comparing, next POI prediction, and spatio-temporal modeling.

---

## 🎯 Project Overview

This repository contains experimental implementations and research explorations in location intelligence, focusing on:

- **Sequential modeling** for next POI prediction using Transformer architectures
- **Multi-modal embeddings** combining POI, temporal, and spatial features
- **Spatio-temporal modeling** for trajectory analysis with attention mechanisms
- **Language models** adapted for location sequences (GPT-2 for shop sequences)
- **Synthetic data generation** for mobility patterns with realistic constraints

---

## 🎯 Experimental Learning Goals

**Note**: This repository is strictly for experimental learning purposes and is not intended for production use. The following objectives represent educational explorations in location intelligence and predictive modeling:

### 1. **Foot-Traffic Projection**
Implemented a Prophet model with a focus on seasonality analysis to predict quarterly foot traffic for major retail chains. This experiment explores time-series forecasting techniques and seasonal pattern recognition in retail analytics.

### 2. **Sales Forecasting** 
Developed custom predictive ML models tailored to each client's industry, metrics, and data, predicting sales for new stores across the United States based on location characteristics. This research focuses on location-based feature engineering and industry-specific modeling approaches.

### 3. **Marketing Research**
Conducted a study on the incremental lift effect of billboard signs on visitation during marketing campaigns. The project's challenge involved defining and identifying a reliable control group for comparison with the exposed group. This experiment explores causal inference methodologies in location-based marketing analytics.

> **Disclaimer**: All implementations are experimental proof-of-concepts designed for learning and research purposes only. They are not optimized, validated, or suitable for production environments.

---

## 📊 Key Experiments & Notebooks

### 🛍️ Next POI Prediction Models

#### 1. **Advanced POI Transformer** (`poi_transformer_advanced.ipynb`)
- **Objective**: Implement Transformer architecture for POI sequence prediction
- **Approach**: Multi-modal embedding fusion with self-attention mechanisms
- **Key Features**:
  - **Multi-Modal Embeddings**: POI ID, category, temporal (hour/day/season), and spatial (lat/lon/region)
  - **Self-Attention Benefits**: Long-range dependencies, periodic pattern detection, complex POI relationships
  - **Multiple Output Heads**: Next POI prediction, category prediction, time-to-next-visit regression
  - **Advanced Features**: Temporal positional encoding, causal masking, attention weight analysis
  - **Demonstrated Advantages**: Captures weekly gym visits, monthly haircuts, context-aware predictions
- **Architecture Highlights**:
  - Learnable modality weights for optimal feature fusion
  - Cross-modal attention for richer interactions
  - Non-linear fusion networks with dropout and layer normalization
  - Optimized for Mac M1 with 200 sequences, 491,985 parameters

#### 2. **Basic POI Trajectory Prediction** (`poi_trajectory_prediction_step1.ipynb`)
- **Objective**: Foundation-level Transformer decoder for POI trajectory modeling
- **Approach**: Autoregressive prediction with temporal context integration
- **Features**:
  - **Temporal-aware architecture**: 4 time periods (morning/afternoon/evening/night)
  - **Transition patterns**: Realistic POI visit sequences (home→coffee→office)
  - **Causal masking**: Proper autoregressive behavior for sequence generation
  - **Multi-objective training**: Combined POI and temporal prediction
- **Model Specs**: 141,130 parameters, TransformerDecoder architecture
- **Results**: Achieves 0.74 training loss with realistic pattern learning

#### 3. **GPT-2 for Shop Sequences** (`gpt2_next_shop.ipynb`)
- **Objective**: Adapt GPT-2 for predicting next shopping locations in sequences
- **Approach**: Fine-tune GPT-2 with custom shop tokens and sequential patterns
- **Features**:
  - **Custom tokenization**: Special tokens for shops, categories, and time periods
  - **Pattern learning**: Sequential shop visits with category transitions
  - **Advanced generation**: Temperature sampling, top-k/top-p filtering
  - **Step-by-step analysis**: Logit visualization and probability distributions
  - **Pattern recognition**: Tests model's understanding of shopping sequences

### 🌍 Trajectory & GPS Modeling

#### 4. **Synthetic GPS Trajectories** (`synthetic_gps_trajectories.ipynb`)
- **Objective**: Generate realistic GPS trajectories using real road network topology
- **Approach**: Transformer encoder trained on random walks through Milan road network
- **Features**:
  - **OSMnx integration**: Real road network from OpenStreetMap
  - **Random walk generation**: Network-constrained trajectory creation
  - **Transformer architecture**: 64-dim embeddings, 4-head attention, 2 layers
  - **Realistic constraints**: Follows actual street connectivity
  - **Folium visualization**: Interactive map rendering of generated trajectories
- **Technical Details**: Milano road network, padding-aware training, cross-entropy loss

---

## 🔬 Research Areas

### **1. Location Intelligence**
- Next POI prediction algorithms
- Mobility pattern recognition
- Customer journey analysis
- Location-based recommendation systems

### **2. Transformer Architectures for Location Data**
- Multi-modal embedding fusion (POI, temporal, spatial)
- Self-attention mechanisms for long-range dependencies
- Causal masking for autoregressive prediction
- Multiple prediction heads (POI, category, time)

### **3. Sequential Modeling**
- Transformer architectures for location sequences
- Language model adaptation for location data (GPT-2)
- Sequence-to-sequence prediction with attention
- Temporal pattern recognition

### **4. Synthetic Data Generation**
- Realistic trajectory simulation using road networks
- Data augmentation for mobility patterns
- Privacy-preserving synthetic datasets
- Category-based sequence generation

---

## 🛠️ Technical Stack

- **Deep Learning**: PyTorch, Transformers (Hugging Face)
- **Geospatial**: OSMnx, Folium, NetworkX
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Language Models**: GPT-2, custom tokenization
- **Device Optimization**: Mac M1/MPS backend support

---

## 🚀 Getting Started

### Prerequisites
```bash
conda create -n location-models python=3.11
conda activate location-models
```

### Installation
```bash
git clone https://github.com/fbalicchia/gym-location-models.git
cd gym-location-models
pip install torch transformers
pip install osmnx folium networkx pandas matplotlib seaborn
pip install scikit-learn jupyter
```

### Quick Start
```bash
# Launch Jupyter notebook
jupyter notebook

# Open and run any of the available experiments:
# - poi_transformer_advanced.ipynb (Most comprehensive)
# - poi_trajectory_prediction_step1.ipynb (Basic example)
# - gpt2_next_shop.ipynb (Language model approach)
# - synthetic_gps_trajectories.ipynb (GPS trajectory generation)
```

---

## 📁 Repository Structure

```
gym-location-model/
├── poi_transformer_advanced.ipynb        # Advanced multi-modal POI Transformer
├── poi_trajectory_prediction_step1.ipynb # Basic POI trajectory prediction
├── gpt2_next_shop.ipynb                  # GPT-2 for shop sequences
├── synthetic_gps_trajectories.ipynb      # GPS trajectory generation
├── synthetic_trajectory.html             # Generated trajectory visualization
└── README.md                             # Project documentation
```

---

## 🔮 Future Directions

- **Multi-modal mobility modeling** (walking, driving, public transport)
- **Privacy-preserving** location analytics
- **Real-time prediction** systems
- **Cross-domain transfer learning** for location models
- **Federated learning** for distributed location data

---

## 📚 Research References

This work draws inspiration from research in:
- Location-based social networks
- Mobility data mining
- Sequential recommendation systems
- Language models for structured sequences

### 📖 **Detailed Literature Review**

For a comprehensive analysis of the academic landscape, including:
- **Transformer-based POI prediction models** (STAN, AutoMTN, GETNext, NextMove)
- **Survey papers and literature evolution** from RNNs to Transformers
- **Comparative analysis** of this repository's contributions vs. existing research
- **Research gaps and future opportunities** in location intelligence
- **Essential reading list** and publication strategies

**➡️ See [research_papers.md](research_papers.md)** for detailed academic context and literature review.

---

## 🤝 Contributing

This is a research playground - feel free to:
- Add new experiments and models
- Improve existing implementations
- Share interesting findings
- Propose new use cases

