import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import precision_recall_fscore_support


def calculate_metrics(predicted_path: str, true_path: str):
    # -----------------------------
    # Load + merge
    # -----------------------------
    predicted_df = pd.read_csv(predicted_path)
    truth_df = pd.read_csv(true_path)

    merged_df = pd.merge(predicted_df, truth_df, on="image_id", suffixes=("_predicted", "_truth"))

    # -----------------------------
    # Helpers
    # -----------------------------
    def parse_cards(card_string):
        if not isinstance(card_string, str) or not card_string.strip():
            return []
        return [c.strip() for c in card_string.split(";") if c.strip()]

    card_cols = [f"player_{i}_cards" for i in range(1, 5)]

    for col in card_cols:
        merged_df[f"{col}_pred_list"] = merged_df[f"{col}_predicted"].apply(parse_cards)
        merged_df[f"{col}_truth_list"] = merged_df[f"{col}_truth"].apply(parse_cards)

    # -----------------------------
    # Exact match metrics
    # -----------------------------
    key_cols = ["active_player", "center_card"]

    for col in key_cols:
        merged_df[f"{col}_match"] = merged_df[f"{col}_predicted"] == merged_df[f"{col}_truth"]

    center_card_accuracy = merged_df["center_card_match"].mean()
    active_player_accuracy = merged_df["active_player_match"].mean()

    # -----------------------------
    # Card-level F1
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
        image_f1_scores.append((2 * tp / denom) if denom > 0 else 0.0)

    overall_f1 = float(np.mean(image_f1_scores))

    # -----------------------------
    # Final score
    # -----------------------------
    final_score = 0.1 * center_card_accuracy + 0.1 * active_player_accuracy + 0.8 * overall_f1

    # -----------------------------
    # Misclassification analysis
    # -----------------------------
    misclassified_fp, misclassified_fn = [], []

    for _, row in merged_df.iterrows():
        for p in range(1, 5):
            truth = row[f"player_{p}_cards_truth_list"]
            pred = row[f"player_{p}_cards_pred_list"]

            for c in set(truth):
                if c not in pred:
                    misclassified_fn.append(c)

            for c in set(pred):
                if c not in truth:
                    misclassified_fp.append(c)

    truth_card_counts = Counter()
    for cards in merged_df[[f"player_{p}_cards_truth_list" for p in range(1, 5)]].values.flatten():
        truth_card_counts.update(cards)

    miss_stats = []
    for card, total in truth_card_counts.items():
        fn = misclassified_fn.count(card)
        miss_stats.append((card, fn, total, (fn / total * 100) if total else 0))

    miss_stats.sort(key=lambda x: x[3], reverse=True)

    top_misclassified = {
        "miss_stats": miss_stats[:11],
        "top_fn": Counter(misclassified_fn).most_common(5),
        "top_fp": Counter(misclassified_fp).most_common(5),
    }

    # -----------------------------
    # Detection count stats
    # -----------------------------
    correct = under = over = 0
    total_cells = len(merged_df) * 4

    for _, row in merged_df.iterrows():
        for p in range(1, 5):
            t = row[f"player_{p}_cards_truth_list"]
            pr = row[f"player_{p}_cards_pred_list"]

            if len(t) == len(pr):
                correct += 1
            elif len(t) > len(pr):
                under += 1
            else:
                over += 1

    count_detection = {
        "accuracy": correct / total_cells,
        "under": under,
        "over": over
    }

    # -----------------------------
    # Colour / symbol metrics
    # -----------------------------
    COLOUR_PREFIXES = {"r", "g", "b", "y"}

    def parse_colour(card):
        if not card:
            return None
        prefix = card.split("_")[0]
        return prefix if prefix in COLOUR_PREFIXES else "black"

    def parse_symbol(card):
        if not card:
            return None
        parts = card.split("_", 1)
        return parts[1] if parts[0] in COLOUR_PREFIXES else card

    def attribute_lists(card_list, fn):
        return [fn(c) for c in card_list if c]

    for attr, fn in [("colour", parse_colour), ("symbol", parse_symbol)]:
        merged_df[f"{attr}_truth"] = merged_df[
            [f"player_{p}_cards_truth_list" for p in range(1, 5)]
        ].apply(lambda r: [x for lst in r for x in attribute_lists(lst, fn)], axis=1)

        merged_df[f"{attr}_pred"] = merged_df[
            [f"player_{p}_cards_pred_list" for p in range(1, 5)]
        ].apply(lambda r: [x for lst in r for x in attribute_lists(lst, fn)], axis=1)

    def mean_f1(tcol, pcol):
        scores = []
        for _, row in merged_df.iterrows():
            G = Counter(row[tcol])
            P = Counter(row[pcol])
            tp = sum((G & P).values())
            fp = sum((P - G).values())
            fn = sum((G - P).values())
            denom = 2 * tp + fp + fn
            scores.append((2 * tp / denom) if denom > 0 else 0.0)
        return float(np.mean(scores))

    def mean_acc(tcol, pcol):
        scores = []
        for _, row in merged_df.iterrows():
            G = Counter(row[tcol])
            P = Counter(row[pcol])
            correct = sum((G & P).values())
            total = sum(G.values())
            scores.append((correct / total) if total > 0 else 0.0)
        return float(np.mean(scores))

    colour_f1 = mean_f1("colour_truth", "colour_pred")
    symbol_f1 = mean_f1("symbol_truth", "symbol_pred")

    colour_acc = mean_acc("colour_truth", "colour_pred")
    symbol_acc = mean_acc("symbol_truth", "symbol_pred")

    # -----------------------------
    # Breakdown metrics
    # -----------------------------
    def build_flat(truth_col, pred_col):
        all_t, all_p = [], []
        for _, row in merged_df.iterrows():
            G = Counter(row[truth_col])
            P = Counter(row[pred_col])
            labels = set(G) | set(P)

            for l in labels:
                g, p = G[l], P[l]
                n = max(g, p)
                all_t.extend([l] * g + ["__extra__"] * (n - g))
                all_p.extend([l] * p + ["__missing__"] * (n - p))
        return all_t, all_p

    def per_class(truth_col, pred_col):
        t, p = build_flat(truth_col, pred_col)
        labels = sorted(set(t) - {"__extra__", "__missing__"})

        pr, re, f1, sup = precision_recall_fscore_support(
            t, p, labels=labels, zero_division=0
        )

        return pd.DataFrame({
            "label": labels,
            "precision": pr,
            "recall": re,
            "f1": f1,
            "support": sup
        }).sort_values("f1")

    colour_breakdown = per_class("colour_truth", "colour_pred")
    symbol_breakdown = per_class("symbol_truth", "symbol_pred")

    # -----------------------------
    # Print structured results
    # -----------------------------

    print("\n=== Evaluation Metrics ===")
    print(f"Center Card Accuracy   : {center_card_accuracy:.4f}")
    print(f"Active Player Accuracy : {active_player_accuracy:.4f}")
    print(f"Overall Card F1 Score  : {overall_f1:.4f}")
    print(f"Overall Evaluation Score: {final_score:.4f}")

    print("\n=== Count Detection ===")
    print(f"Exact Count Accuracy: {count_detection['accuracy']:.4f}")
    print(f"Under-detection     : {count_detection['under']}")
    print(f"Over-detection      : {count_detection['over']}")

    print("\n=== Colour Metrics ===")
    print(f"Colour F1 Score    : {colour_f1:.4f}")
    print(f"Colour Accuracy    : {colour_acc:.4f}")
    print(colour_breakdown)

    print("\n=== Symbol Metrics ===")
    print(f"Symbol F1 Score    : {symbol_f1:.4f}")
    print(f"Symbol Accuracy    : {symbol_acc:.4f}")
    print(symbol_breakdown)

    print("\n=== Top Misclassified Cards ===")
    print("Card     | False Negatives | Total Occurrences | % Missed")
    for card, fn, total, pct in top_misclassified["miss_stats"]:
        print(f"{card:10} {fn:15} {total:19} {pct:9.2f}%")

    print("\nTop 5 False Negatives:")
    for card, count in top_misclassified["top_fn"]:
        print(f"{card:10} {count}")
    print("\nTop 5 False Positives:")
    for card, count in top_misclassified["top_fp"]:
        print(f"{card:10} {count}")
    


if __name__ == "__main__": 
    train_submission_path = "data/output/train_submission_2026-05-20-15-30-38.csv"
    calculate_metrics(train_submission_path, "data/train.csv")