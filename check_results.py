import pandas as pd
import numpy as np
from collections import Counter

predicted_path = "data/output/train_submission_2026-05-20-09-53-19.csv"
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

        # False Negatives: card exists in truth but not predicted
        for card in set(truth_cards):
            if card not in pred_cards:
                misclassified_fn.append(card)

        # False Positives: card predicted but not in truth
        for card in set(pred_cards):
            if card not in truth_cards:
                misclassified_fp.append(card)

# Count total true occurrences
truth_card_counts = Counter()
for cards in merged_df[[f"player_{p}_cards_truth_list" for p in range(1, 5)]].values.flatten():
    truth_card_counts.update(cards)

# Compute miss statistics
miss_stats = []

for card, total_count in truth_card_counts.items():
    fn_count = misclassified_fn.count(card)
    miss_rate = (fn_count / total_count) * 100 if total_count > 0 else 0
    miss_stats.append((card, fn_count, total_count, miss_rate))

# Sort by relative miss rate (descending)
miss_stats.sort(key=lambda x: x[3], reverse=True)

print("\nMisclassified Cards (False Negatives) sorted by Miss Rate:")
for card, fn_count, total_count, miss_rate in miss_stats[:11]:
    print(f"{card}: {fn_count} missed out of {total_count} occurrences ({miss_rate:.2f}%)")

# Print top misclassified cards

i = 5
print(f"\nTop {i} Most Commonly Misclassified Cards:")
print(f"\nFalse Negatives (missed cards):")
for card, count in Counter(misclassified_fn).most_common(i):
    print(f"{card}: {count}")
print(f"\nFalse Positives (incorrectly predicted cards):")
for card, count in Counter(misclassified_fp).most_common(i):
    print(f"{card}: {count}")



# -----------------------------
# 7. Check number of cards detected per player
# -----------------------------

# check if the number of cards detected matches the truth
# print average, number of correct, under and over detections in total
total_images = len(merged_df)
correct_detections = 0
under_detections = 0
over_detections = 0 

for _, row in merged_df.iterrows():
    for player in range(1, 5):
        truth_cards = row[f"player_{player}_cards_truth_list"]
        pred_cards = row[f"player_{player}_cards_pred_list"]

        if len(truth_cards) == len(pred_cards):
            correct_detections += 1
        elif len(truth_cards) > len(pred_cards):
            under_detections += 1
        else:
            over_detections += 1   

print(f"\nCard Count Detection Accuracy: {correct_detections / (total_images * 4):.4f}")
print(f"Under Detections: {under_detections} ({under_detections / (total_images * 4):.4f})")
print(f"Over Detections: {over_detections} ({over_detections / (total_images * 4):.4f})")

# -----------------------------
# 8. Colour and symbol metrics
# -----------------------------
COLOUR_PREFIXES = {"r", "g", "b", "y"}

def parse_colour(card: str):
    if not card:
        return None
    prefix = card.split("_")[0]
    return prefix if prefix in COLOUR_PREFIXES else "black"

def parse_symbol(card: str):
    if not card:
        return None
    parts = card.split("_", 1)
    return parts[1] if parts[0] in COLOUR_PREFIXES else card

def attribute_lists(card_list: list[str], fn) -> list[str]:
    return [fn(c) for c in card_list if c]

# Aggregate ground-truth and predicted attributes across all four players per image
for attr, fn in [("colour", parse_colour), ("symbol", parse_symbol)]:
    merged_df[f"{attr}_truth"] = merged_df[
        [f"player_{p}_cards_truth_list" for p in range(1, 5)]
    ].apply(lambda row: [x for lst in row for x in attribute_lists(lst, fn)], axis=1)

    merged_df[f"{attr}_pred"] = merged_df[
        [f"player_{p}_cards_pred_list" for p in range(1, 5)]
    ].apply(lambda row: [x for lst in row for x in attribute_lists(lst, fn)], axis=1)

# Per-image F1 (multiset-aware, identical logic to section 4)
def mean_f1(truth_col: str, pred_col: str) -> tuple[float, list[float]]:
    scores = []
    for _, row in merged_df.iterrows():
        G = Counter(row[truth_col])
        P = Counter(row[pred_col])
        tp = sum((G & P).values())
        fp = sum((P - G).values())
        fn = sum((G - P).values())
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)), scores

colour_f1, _ = mean_f1("colour_truth", "colour_pred")
symbol_f1, _ = mean_f1("symbol_truth", "symbol_pred")

print(f"Colour F1 Score: {colour_f1:.4f}")
print(f"Symbol F1 Score: {symbol_f1:.4f}")

# Per-image accuracy (fraction of attributes exactly matched, multiset-aware)
def mean_accuracy(truth_col: str, pred_col: str) -> float:
    scores = []
    for _, row in merged_df.iterrows():
        G = Counter(row[truth_col])
        P = Counter(row[pred_col])
        correct = sum((G & P).values())
        total = sum(G.values())
        scores.append((correct / total) if total > 0 else 0.0)
    return float(np.mean(scores))

colour_acc = mean_accuracy("colour_truth", "colour_pred")
symbol_acc = mean_accuracy("symbol_truth", "symbol_pred")

print(f"Colour Accuracy: {colour_acc:.4f}")
print(f"Symbol Accuracy: {symbol_acc:.4f}")


# -----------------------------
# 9. Per-colour and per-symbol breakdown
# -----------------------------
from sklearn.metrics import precision_recall_fscore_support

def build_flat_lists(truth_col: str, pred_col: str) -> tuple[list[str], list[str]]:
    """Flatten all per-image attribute lists into two aligned lists (padded with '__missing__' / '__extra__')."""
    all_truth, all_pred = [], []
    for _, row in merged_df.iterrows():
        G = Counter(row[truth_col])
        P = Counter(row[pred_col])
        all_labels = set(G) | set(P)
        for label in all_labels:
            g_count = G[label]
            p_count = P[label]
            n = max(g_count, p_count)
            all_truth.extend([label] * g_count + ["__extra__"] * (n - g_count))
            all_pred.extend([label] * p_count + ["__missing__"] * (n - p_count))
    return all_truth, all_pred

def per_class_metrics(truth_col: str, pred_col: str) -> pd.DataFrame:
    truth_flat, pred_flat = build_flat_lists(truth_col, pred_col)
    labels = sorted(set(truth_flat) - {"__extra__", "__missing__"})
    p, r, f1, support = precision_recall_fscore_support(
        truth_flat, pred_flat, labels=labels, zero_division=0
    )
    return pd.DataFrame({
        "label":     labels,
        "precision": p.round(4),
        "recall":    r.round(4),
        "f1":        f1.round(4),
        "support":   support,
    }).sort_values("f1", ascending=True)

colour_breakdown = per_class_metrics("colour_truth", "colour_pred")
symbol_breakdown = per_class_metrics("symbol_truth", "symbol_pred")

print("\nPer-colour metrics:")
print(colour_breakdown.to_string(index=False))

print("\nPer-symbol metrics:")
print(symbol_breakdown.to_string(index=False))