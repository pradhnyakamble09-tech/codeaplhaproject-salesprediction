"""
Generates a realistic advertising-spend-to-sales dataset in the shape
commonly used for this task (similar to the classic "Advertising.csv"
dataset used in many Sales Prediction tutorials): TV, Radio, and
Newspaper advertising spend, plus target segment and sales platform,
with Sales computed from a genuine (noisy, diminishing-returns) response
model so the relationships are learnable.

The CodeAlpha task PDF's "DOWNLOAD DATASET FROM here" link is a bare
placeholder with no resolvable URL, so this script builds a realistic
stand-in. Swap in a real dataset at data/advertising.csv with matching
column names to use real data instead — no code changes needed as long
as the target column stays "Sales".
"""

import numpy as np
import pandas as pd

np.random.seed(11)
N = 500

segments = ["Young Adults", "Families", "Seniors", "Professionals"]
platforms = ["Online", "TV Network", "Retail Partner", "Social Media"]

rows = []
for _ in range(N):
    tv = np.clip(np.random.exponential(80), 0, 300)
    radio = np.clip(np.random.exponential(20), 0, 50)
    newspaper = np.clip(np.random.exponential(15), 0, 60)

    segment = np.random.choice(segments)
    platform = np.random.choice(platforms)

    # Diminishing-returns response: sqrt scaling on spend (more spend
    # helps, but with decreasing marginal benefit), channel effectiveness
    # ordered TV > Radio > Newspaper, matching real-world ad ROI patterns
    segment_multiplier = {"Young Adults": 1.10, "Families": 1.00, "Seniors": 0.85, "Professionals": 1.05}[segment]
    platform_multiplier = {"Online": 1.08, "TV Network": 1.00, "Retail Partner": 0.95, "Social Media": 1.12}[platform]

    base_sales = 4.0
    sales = base_sales
    sales += 0.85 * np.sqrt(tv)
    sales += 0.55 * np.sqrt(radio) * 2
    sales += 0.12 * np.sqrt(newspaper)
    sales *= segment_multiplier * platform_multiplier
    sales += np.random.normal(0, 1.0)
    sales = max(0.5, sales)

    rows.append({
        "TV": round(tv, 2),
        "Radio": round(radio, 2),
        "Newspaper": round(newspaper, 2),
        "Target_Segment": segment,
        "Platform": platform,
        "Sales": round(sales, 2),
    })

df = pd.DataFrame(rows)
df.to_csv("data/advertising.csv", index=False)
print(df.shape)
print(df.head())
