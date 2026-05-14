import pandas as pd
import numpy as np
from collections import Counter

predicted_path = "data/train_submission_2026-05-14-14-56-55.csv"
truth_path = "data/train.csv"

predicted_df = pd.read_csv(predicted_path)
truth_df = pd.read_csv(truth_path)

# Merge
merged_df = pd.merge(
    predicted_df,
    truth_df,
    on="image_id",
    suffixes=("_predicted", "_truth")
)

# -----------------------------
# 1. Parse card strings once
# -----------------------------
def parse_cards(card_string):
    if not isinstance(card_string, str) or not card_string.strip():
        return []
    return [c.strip() for c in card_string.split(";") if c.strip()]

card_cols = [
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
]

for col in card_cols:
    merged_df[f"{col}_pred_list"] = merged_df[f"{col}_predicted"].apply(parse_cards)
    merged_df[f"{col}_truth_list"] = merged_df[f"{col}_truth"].apply(parse_cards)

# -----------------------------
# 2. Exact match metrics
# -----------------------------
key_cols = ["active_player", "center_card"]

for col in key_cols:
    merged_df[f"{col}_match"] = merged_df[f"{col}_predicted"] == merged_df[f"{col}_truth"]

center_card_accuracy = merged_df["center_card_match"].mean()
active_player_accuracy = merged_df["active_player_match"].mean()

print(f"Center Card Accuracy: {center_card_accuracy:.4f}")
print(f"Active Player Accuracy: {active_player_accuracy:.4f}")


# -----------------------------
# 4. Card-level F1 evaluation
# -----------------------------
image_f1_scores = []

for _, row in merged_df.iterrows():
    tp = fp = fn = 0

    for p in range(1, 5):
        g = row[f"player_{p}_cards_truth_list"]
        pred = row[f"player_{p}_cards_pred_list"]

        G = Counter(g)
        P = Counter(pred)

        tp += sum((G & P).values())
        fp += sum((P - G).values())
        fn += sum((G - P).values())

    denom = 2 * tp + fp + fn
    f1 = (2 * tp / denom) if denom > 0 else 0.0
    image_f1_scores.append(f1)

overall_f1_score = np.mean(image_f1_scores)

print(f"Overall Card F1 Score: {overall_f1_score:.4f}")

# -----------------------------
# 5. Final weighted score
# -----------------------------
final_score = (
    0.1 * center_card_accuracy +
    0.1 * active_player_accuracy +
    0.8 * overall_f1_score
)

print(f"Overall Evaluation Score: {final_score:.4f}")

# -----------------------------
# 6. Misclassified card analysis (clean)
# -----------------------------
misclassified_fp = []
misclassified_fn = []

for _, row in merged_df.iterrows():
    for player in range(1, 5):
        truth_cards = row[f"player_{player}_cards_truth_list"]
        pred_cards = row[f"player_{player}_cards_pred_list"]

        # Count misclassified cards (in truth but not in pred)
        for card in set(truth_cards):
            if card not in pred_cards:
                misclassified_fn.append(card)

        # Count misclassified cards (in pred but not in truth)
        for card in set(pred_cards):
            if card not in truth_cards:
                misclassified_fp.append(card)

i = 5
print(f"\nTop {i} Most Commonly Misclassified Cards:")
print(f"\nFalse Negatives (missed cards):")
for card, count in Counter(misclassified_fn).most_common(i):
    print(f"{card}: {count}")
print(f"\nFalse Positives (incorrectly predicted cards):")
for card, count in Counter(misclassified_fp).most_common(i):
    print(f"{card}: {count}")