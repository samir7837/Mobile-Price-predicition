# 📱 Mobile Price Prediction using Machine Learning

## 🧭 Project Overview

This project aims to **predict the price range of mobile phones** based on their specifications such as RAM, battery power, display resolution, and more.
By analyzing key features and training multiple classification models, we determine which specifications most influence the pricing tier of mobile devices.

This work was developed as part of an **internship project** to demonstrate end-to-end data analysis, feature engineering, model training, and evaluation.

---

## 🧩 Dataset

The dataset contains **2000 samples and 20 features** with the target column `price_range`:

* `0` → Low cost
* `1` → Medium cost
* `2` → High cost
* `3` → Very high cost

Each row represents a unique mobile phone with attributes like:

* `battery_power` — Total energy a battery can store (mAh)
* `ram` — Memory size (MB)
* `px_height`, `px_width` — Screen resolution
* `mobile_wt` — Weight of the mobile phone (grams)
* `talk_time`, `n_cores`, `int_memory`, etc.

---

## ⚙️ Technologies Used

* **Python 3.10+**
* **Libraries:**
  `pandas`, `numpy`, `matplotlib`, `seaborn`,
  `scikit-learn`, `xgboost`, `lightgbm`, `joblib`

---

## 🧠 Project Workflow

1. **Data Loading & Exploration**

   * Loaded CSV data and verified class balance.
   * Explored distributions and relationships using visualizations (Seaborn & Matplotlib).

2. **Preprocessing**

   * Feature scaling using `StandardScaler`.
   * Stratified train-test split (80:20).

3. **Model Training**

   * Trained multiple models:

     * Logistic Regression
     * Decision Tree
     * Random Forest
     * XGBoost
     * LightGBM

4. **Model Evaluation**

   * Compared models on accuracy and classification reports.
   * Visualized confusion matrices and feature importances.

5. **Model Saving**

   * Saved the best-performing model (`Logistic Regression`) using `joblib` for reuse.

---

## 📊 Results & Insights

| Model               | Accuracy    | Remarks                                         |
| ------------------- | ----------- | ----------------------------------------------- |
| Logistic Regression | ⭐ Highest   | Best generalization, dataset linearly separable |
| Decision Tree       | Good        | Slight overfitting tendency                     |
| Random Forest       | Strong      | High accuracy but slower                        |
| XGBoost             | Competitive | Similar to RF but not better                    |
| LightGBM            | Competitive | Performs well but simple model wins             |

**Key Findings:**

* `ram`, `battery_power`, and `px_height` are the most influential features.
* The dataset is clean and balanced, favoring linear models.
* Logistic Regression achieved the best tradeoff between simplicity and performance.

---

## 🧾 How to Run

1. **Clone this repository**

   ```bash
   git clone https://github.com/yourusername/mobile-price-prediction.git
   cd mobile-price-prediction
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**

   ```bash
   jupyter notebook Mobile-Price-Prediction.ipynb
   ```

4. **(Optional)** Run predictions using the saved model:

   ```python
   import joblib, pandas as pd
   model = joblib.load("best_logistic_regression_mobile_price.joblib")
   scaler = joblib.load("price_scaler.joblib")

   # Example new phone
   sample = pd.DataFrame([{
       "battery_power": 1000, "ram": 2000, "px_height": 800, "px_width": 1200, "mobile_wt": 150, ...
   }])
   pred = model.predict(scaler.transform(sample))
   print("Predicted price range:", pred[0])
   ```

---

## 🚀 Future Improvements

* Add **cross-validation** and **SHAP** explainability plots.
* Deploy model using **Streamlit** for interactive predictions.
* Automate retraining pipeline for new data.

---

## 👨‍💻 Author

**Samir Sharma**

## 🏁 Conclusion

This project demonstrates a full machine learning pipeline — from exploration to deployment — and shows that **simple, well-regularized models can outperform complex ones** when data is clean and structured.

