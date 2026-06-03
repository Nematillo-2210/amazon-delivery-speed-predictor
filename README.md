# Amazon Delivery Speed Predictor

XGBoost-powered Streamlit app that predicts whether an Amazon delivery will be fast or slow. Trained on Indian Amazon delivery data with SHAP explainability and live deployment on Hugging Face Spaces.

🚀 **[Live App](https://huggingface.co/spaces/nematillo-2210/shipment-delay-prediction)**

---

## Model Card

### Section 1: Model Description
- **Algorithm:** XGBoost (Extreme Gradient Boosting) Classifier
- **Input Features (15 total):**
  - *Agent Profile:* `Agent_Age`, `Agent_Rating`
  - *Geospatial:* `Store_Latitude`, `Store_Longitude`, `Drop_Latitude`, `Drop_Longitude`, `Area`
  - *Order Details:* `Category`
  - *Temporal:* `day_of_week`, `is_weekend`, `month`, `order_hour_block`
  - *Environmental & Logistics:* `Weather`, `Traffic`, `Vehicle`

### Section 2: Intended Use
Designed solely to predict binary delivery speed — Fast (1) vs. Slow (0) — for Amazon shipments in India. The exact definition of "Fast" is not documented in the source dataset but reflects deliveries completed ahead of the estimated arrival threshold.

### Section 3: Out-of-Scope Use
Any application outside operational shipment speed forecasting is out of scope. This model must not be used to evaluate agent performance, determine bonuses, or inform hiring and firing decisions.

### Section 4: Evaluation Results
- Training accuracy: 84.63% | Test accuracy: 82.38% — minimal overfitting
- Test set: 8,748 samples

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Slow (0) | 80% | 85% | 82% |
| Fast (1) | 85% | 80% | 82% |

Confusion matrix: 3,619 TN · 650 FP · 891 FN · 3,588 TP

### Section 5: Limitations
- Trained exclusively on Indian Amazon delivery data. Performance will degrade significantly outside this geographic and operational context.
- 891 false negatives — the model incorrectly labels fast deliveries as slow, particularly for high-rated agents in poor conditions or low-rated agents in optimal conditions.

### Section 6: Ethical Considerations
The dataset reflects a systemic pattern where lower-rated and younger agents recorded faster deliveries more frequently. The model learns and replicates this bias. If misapplied to personnel decisions, it will unfairly penalize experienced or safety-conscious agents whose ratings reflect priorities other than raw speed.

---

## How To Use

```python
import joblib
model = joblib.load('best_model.pkl')
predictions = model.predict(X_new)
```

---

## Tech Stack
- Python, XGBoost, scikit-learn, SHAP
- Streamlit for the web interface
- Deployed on Hugging Face Spaces
