# 🧠 Deep Learning Sentiment Analysis

**LSTM-based** text sentiment classifier — Positive 😊 / Neutral 😐 / Negative 😞

---

## 📁 Project Structure

```
Sentiment_analysis/
├── data/
│   └── dataset.csv          # Labeled training data (~240 sentences)
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Text cleaning, tokenization, padding
│   └── model.py             # LSTM model architecture
├── templates/
│   └── index.html           # Web UI
├── saved_model/             # Created after training
│   ├── model.h5
│   ├── tokenizer.pkl
│   └── config.pkl
├── train.py                 # Train the model
├── predict.py               # CLI prediction
├── app.py                   # Flask web server
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
```

This will:
- Load & preprocess the dataset
- Build the LSTM model
- Train for 20 epochs
- Save the model to `saved_model/`

### 3. Test via CLI

```bash
python predict.py
```

### 4. Launch Web UI

```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🏗️ Model Architecture

```
Embedding (128-dim) → LSTM (64 units) → Dense (32, ReLU) → Dropout (0.5) → Softmax (3 classes)
```

| Layer        | Details                          |
|-------------|----------------------------------|
| Embedding   | Vocab × 128 dimensions           |
| LSTM        | 64 hidden units                  |
| Dense       | 32 units, ReLU activation        |
| Dropout     | 50% dropout for regularization   |
| Output      | 3 classes, Softmax activation    |

---

## 🛠️ Technologies

| Component           | Technology           |
|---------------------|----------------------|
| Language            | Python               |
| Deep Learning       | TensorFlow / Keras   |
| Model               | LSTM                 |
| Data Processing     | Pandas, NumPy        |
| Web Framework       | Flask                |

---

## 📊 Example

**Input:** `"I love this phone it is amazing"`

**Output:**
```
Sentiment  : Positive 😊
Confidence : 94.2%
```
