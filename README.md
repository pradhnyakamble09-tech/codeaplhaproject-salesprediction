# CodeAlpha_SalesPrediction

**CodeAlpha Data Science Internship — Task 4: Sales Prediction using Python**

## 📖 Overview
This project predicts sales based on advertising spend across TV, Radio,
and Newspaper channels, plus target customer segment and sales platform,
then analyzes how changes in advertising spend impact sales outcomes to
deliver actionable marketing insights.

## ⚠️ About the dataset
The CodeAlpha task PDF's "DOWNLOAD DATASET FROM here" link is a bare
placeholder with no resolvable URL. `generate_dataset.py` builds a
**realistic stand-in** (500 records) where Sales is generated from a
genuine diminishing-returns response model (`sales ≈ base + a·√TV +
b·√Radio + c·√Newspaper`, scaled by segment/platform effects, plus
market noise) — the same kind of relationship real advertising-response
data shows, so the models learn a real signal rather than noise.

**To use a real dataset instead** (e.g. the classic `Advertising.csv`
dataset with TV/Radio/Newspaper/Sales columns): save it as
`data/advertising.csv` with matching column names (add `Target_Segment`
and `Platform` columns, or remove those two lines from the feature list
in `sales_prediction.py` if your data doesn't have them) and re-run —
no other code changes needed.

## 📂 Project Structure
```
CodeAlpha_SalesPrediction/
├── generate_dataset.py     # Builds data/advertising.csv
├── sales_prediction.py     # Main script: EDA, training, evaluation, impact analysis
├── requirements.txt
├── results.json            # Model metrics + advertising impact, generated on run
├── data/
│   └── advertising.csv
├── plots/
│   ├── spend_vs_sales.png
│   ├── correlation_heatmap.png
│   ├── sales_by_segment.png
│   ├── sales_by_platform.png
│   ├── model_comparison.png
│   ├── actual_vs_predicted.png
│   └── advertising_impact.png
└── models/
    └── best_model.pkl       # Full sklearn pipeline (preprocessing + model)
```

## ⚙️ Setup
```bash
pip install -r requirements.txt
```

## ▶️ Run
```bash
python generate_dataset.py        # only needed if data/advertising.csv doesn't already exist
python sales_prediction.py
```

## 📊 Results
| Model | R² | RMSE | MAE |
|---|---|---|---|
| Linear Regression | 0.864 | 1.496 | 1.203 |
| Ridge Regression | 0.864 | 1.496 | 1.201 |
| Random Forest | 0.855 | 1.545 | 1.250 |
| **Gradient Boosting** | **0.879** | **1.408** | **1.154** |

**Best model: Gradient Boosting** — R² = 0.879 on the held-out test set.

### Advertising impact analysis (from median baseline, +20% spend increase)
| Channel | Estimated Sales Lift |
|---|---|
| **TV** | **+1.78** |
| Radio | +0.39 |
| Newspaper | −0.02 (negligible) |

**Marketing takeaway:** TV advertising drives by far the largest
incremental return on spend, Radio has a smaller but positive effect,
and Newspaper spend shows essentially no measurable impact on sales in
this data — budget is better reallocated toward TV and Radio.

## 🧠 Key Concepts Demonstrated
- Data cleaning and preprocessing
- Preprocessing pipeline: `StandardScaler` + `OneHotEncoder` via `ColumnTransformer`
- Training and comparing regression algorithms (linear + ensemble)
- Evaluation: R², RMSE, MAE, 5-fold cross-validation
- Scenario simulation to quantify marginal impact of spend changes per channel
- Turning a model into a business-actionable recommendation

## ✅ CodeAlpha Submission Checklist
- [x] Source code uploaded to a GitHub repo named `CodeAlpha_SalesPrediction`
- [ ] Post project status update on LinkedIn, tagging **@CodeAlpha**
- [ ] Post a video walkthrough of the project on LinkedIn with the GitHub repo link
- [ ] Submit the completed task via the official Submission Form (shared in the WhatsApp group)

---
*Built for the CodeAlpha Data Science Internship program.*
