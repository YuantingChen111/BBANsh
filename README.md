# BBANsh
This project is developed based on **BERT** (Bidirectional Encoder Representations from Transformers) and **BAN** (Bilinear Attention Network), aiming to predict potent shRNA (short hiarpin RNA) by integrating pretrained sequence embeddings and attention-based feature fusion.

---

## 📁 Project Structure
```text
project_root/
├── external_data/              # Input or auxiliary data
│   └── external.csv
├── pretrained_model/           # Pretrained BERT model files
│   ├── gena_lm_bert_base_t2t/
│   └── DNABERT_6/
├── results/                     # Model outputs, checkpoints, predictions
│   ├── roc/
│   ├── prc/
│   └── metrics/
├── train.py                # Training and prediction scripts
├── predict.py
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview and usage instructions
```
---

## ⚙️ Environment Setup
`python
pip install -r requirements.txt
`
---

## 🔽 Download Pretrained BERT Model
recommend source
DNABERT:https://huggingface.co/zhihan1996/DNA_bert_6/tree/main
GENA-LM:https://huggingface.co/AIRI-Institute/gena-lm-bert-base-t2t

---

## 🏋️ Model Training
Use the training script to train the model:

python train.py

---

## 🔍 Model Prediction
Use the best model to make predictions on test data:
python predict.py

---
