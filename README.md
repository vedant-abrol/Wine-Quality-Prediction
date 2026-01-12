# 🍷 Wine Quality Prediction

<div align="center">

![Apache Spark](https://img.shields.io/badge/Apache%20Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![AWS](https://img.shields.io/badge/AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Predict wine quality from chemical properties using distributed machine learning**

[Getting Started](#-getting-started) •
[Live Demo](#-interactive-demo) •
[How It Works](#-how-it-works) •
[Results](#-results) •
[Docker](#-docker-deployment)

</div>

---

## 🎯 Interactive Demo

<div align="center">

**🍷 Try the beautiful wine-themed web application! 🍷**

</div>

No AWS or cloud setup needed! Run the interactive demo locally to showcase the model:

### Quick Demo Launch

```bash
# Install demo dependencies
pip install streamlit plotly scikit-learn

# Launch the wine-themed GUI
streamlit run wine_demo_app.py
```

Then open **http://localhost:8501** in your browser.

### Demo Features

| Feature | Description |
|---------|-------------|
| 🎨 **Wine-Themed UI** | Beautiful burgundy, wine-red, and gold color scheme |
| 🎚️ **Interactive Sliders** | Adjust all 11 chemical properties in real-time |
| 📊 **Visual Analytics** | Radar charts, probability distributions, and correlations |
| 🤖 **Live Predictions** | Instant quality predictions with confidence scores |
| 📈 **Dataset Explorer** | Browse and analyze the training data |
| 🎯 **Quick Presets** | One-click presets for everyday vs premium wines |

<details>
<summary>📸 Demo Screenshots</summary>

The demo application features:
- **Main Prediction Panel** - Input wine properties and get quality predictions
- **Wine Composition Radar** - Visual profile of your wine's chemical makeup
- **Quality Distribution** - See how predictions compare to the dataset
- **About Section** - Technical details about the ML pipeline

</details>

---

## 📋 Overview

This project uses **Apache Spark MLlib** to train machine learning models that predict wine quality based on 11 chemical properties. The model is trained in parallel across multiple AWS EC2 instances and can be deployed using Docker for easy execution anywhere.

### Key Features

- 🚀 **Distributed Training** - Parallel processing across 5 EC2 instances
- 🎯 **Two Model Comparison** - Logistic Regression vs Random Forest
- 📊 **Cross-Validation** - 3-fold CV for robust evaluation
- 🐳 **Docker Support** - Containerized for easy deployment
- 📝 **Clean Code** - Well-documented and modular

---

## 🏗️ Project Structure

```
Wine-Quality-Prediction/
├── Training.py           # Model training script (Spark MLlib)
├── Predictions.py        # Inference/prediction script
├── wine_demo_app.py      # 🆕 Interactive demo GUI (Streamlit)
├── docker/
│   └── Predictions.py    # Docker-optimized prediction script
├── TrainingDataset.csv   # Training data (1,280 samples)
├── ValidationDataset.csv # Test data (799 samples)
├── Dockerfile            # Container configuration
├── requirements.txt      # Python dependencies
└── README.md             # You are here!
```

---

## 🔬 How It Works

### The Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Load Data  │───▶│  Clean &    │───▶│  Feature    │───▶│   Train     │
│   (CSV)     │    │  Prepare    │    │  Assembly   │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Output    │◀───│  Evaluate   │◀───│   Predict   │◀───│    Save     │
│   Results   │    │   (F1)      │    │  (Test)     │    │   Model     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### Input Features (11 Chemical Properties)

| Feature | Description |
|---------|-------------|
| Fixed Acidity | Tartaric acid content |
| Volatile Acidity | Acetic acid content |
| Citric Acid | Freshness indicator |
| Residual Sugar | Remaining sugar after fermentation |
| Chlorides | Salt content |
| Free SO₂ | Free sulfur dioxide |
| Total SO₂ | Total sulfur dioxide |
| Density | Mass per unit volume |
| pH | Acidity level (0-14 scale) |
| Sulphates | Antimicrobial additive |
| Alcohol | Alcohol percentage |

### Models Compared

| Model | F1 Score | Selected |
|-------|----------|----------|
| **Logistic Regression** | 0.5729 | ✅ |
| Random Forest | 0.5035 | ❌ |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Java 8 or 11 (required for Spark)
- Apache Spark 3.2+

### Quick Start (Local)

**1. Clone the repository**
```bash
git clone https://github.com/vedant-abrol/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Set Java environment** (if not already set)
```bash
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk  # Linux
# or
export JAVA_HOME=$(/usr/libexec/java_home)     # macOS
```

**4. Train the model**
```bash
python Training.py
```

**5. Make predictions**
```bash
python Predictions.py
```

---

## ☁️ AWS EC2 Cluster Setup

For distributed training across multiple machines:

### Step 1: Launch EC2 Instances

| Setting | Value |
|---------|-------|
| Number of instances | 5 |
| AMI | Ubuntu Server 20.04 LTS |
| Instance type | t2.large |
| Storage | 16 GiB |
| Security Group | Allow SSH (port 22) |

### Step 2: Install Dependencies (on each instance)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Java
sudo apt install -y default-jre

# Install Spark
wget -O spark.tgz "https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop2.7.tgz"
sudo mkdir -p /opt/spark
sudo tar -xf spark.tgz -C /opt/spark --strip-components=1
rm spark.tgz

# Add to PATH
echo 'export SPARK_HOME=/opt/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin' >> ~/.bashrc
source ~/.bashrc

# Install Python packages
sudo apt install -y python3-pip
pip3 install -r requirements.txt
```

### Step 3: Configure Spark Cluster

**On Master Node:**
```bash
# Set environment
export SPARK_MASTER_PORT=7077
export SPARK_MASTER_WEBUI_PORT=8090

# Start master
/opt/spark/sbin/start-master.sh
```

**On Worker Nodes:**
```bash
# Connect to master (replace MASTER_IP)
/opt/spark/sbin/start-worker.sh spark://MASTER_IP:7077
```

### Step 4: Run Training

```bash
# On master node
spark-submit --master spark://MASTER_IP:7077 Training.py
```

---

## 🐳 Docker Deployment

### Option 1: Build Locally

```bash
# Build the image
docker build -t wine-quality-predictor .

# Run with local data
docker run -v $(pwd):/data wine-quality-predictor
```

### Option 2: Use Pre-built Image

```bash
# Pull from Docker Hub (replace with your username)
docker pull vedantabrol/wine-quality-predictor

# Run predictions
docker run -v /path/to/your/data:/data vedantabrol/wine-quality-predictor
```

### Docker Commands Reference

| Command | Description |
|---------|-------------|
| `docker build -t wine-quality-predictor .` | Build image |
| `docker run wine-quality-predictor` | Run with default data |
| `docker run -v /data:/data wine-quality-predictor` | Run with custom data |
| `docker push user/wine-quality-predictor` | Push to Docker Hub |

---

## 📊 Results

### Model Performance

```
============================================================
PREDICTION SUMMARY
============================================================
Model:          Logistic Regression
F1 Score:       0.5729
Accuracy:       ~57%
============================================================
```

### Sample Predictions

```
   #  Actual  Predicted  Match
   1.   6        6        ✓
   2.   5        5        ✓
   3.   5        6        ✗
   4.   6        6        ✓
   5.   5        5        ✓
```

---

## 🛠️ Usage Examples

### Basic Training
```bash
python Training.py
```

### Prediction with Custom Data
```bash
python Predictions.py --test-data my_wines.csv
```

### Force Model Retraining
```bash
python Predictions.py --force-retrain
```

### All Options
```bash
python Predictions.py --help

Options:
  --training-data PATH   Training data CSV
  --test-data PATH       Test data CSV  
  --model-path PATH      Saved model directory
  --force-retrain        Retrain even if model exists
```

---

## 📁 Data Format

Your CSV files should use semicolon (`;`) as separator:

```csv
"fixed acidity";"volatile acidity";"citric acid";...;"quality"
7.4;0.7;0;1.9;0.076;11;34;0.9978;3.51;0.56;9.4;5
7.8;0.88;0;2.6;0.098;25;67;0.9968;3.2;0.68;9.8;5
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is part of CS643 Cloud Computing coursework.

---

## 👤 Author

**Vedant Abrol**

- GitHub: [@vedant-abrol](https://github.com/vedant-abrol)

---

<div align="center">

Made with ❤️ using Apache Spark

</div>
