import pandas as pd
import numpy as np
import cv2
import os
from datetime import datetime
from PIL import Image

from src.cropping import get_sector_polygons, extract_sector, assign_cards_to_players
from src.active import detect_active_player
from src.segmentation_border import segmented_cards
from src.classify import classify_card
from src.config import load_config, set_global_config

mode = "test" # test ot train

if mode == "test":
    submission_df = pd.read_csv("data/sample_submission.csv")
elif mode == "train":
    submission_df = pd.read_csv("data/train.csv")
else:
    raise ValueError("Invalid mode. Choose 'test' or 'train'.")
print(submission_df.head())


# remove ID 'L1000867' because it is not in the test set
submission_df = submission_df[submission_df["image_id"] != "L1000867"]
print(len(submission_df))

# Load and set global configuration
config = load_config("config.json")
set_global_config(config)


# iterate over rows of the submission dataframe
for index, row in submission_df.iterrows():

    active_player = "EMPTY"
    center_card = "EMPTY"
    player_cards = ["EMPTY"] * 4

    image_id = row["image_id"]
    print(f"Processing {image_id}...")

    image_path = os.path.join(f"data/{mode}_images", f"{image_id}.jpg")

    # Load image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil_rgb = Image.fromarray(img_rgb)

    active_player = detect_active_player(img_rgb)["active_player"]
    print(f"Active player: {active_player}")

    # Get sector polygons for player regions
    polygons = get_sector_polygons(img_pil_rgb)

    # Detect all cards in the full image with coordinates
    detected_cards_with_coords = segmented_cards(img_rgb, return_coords=True, plot=False)

    # Assign cards to players based on their coordinates
    player_cards_dict = assign_cards_to_players(detected_cards_with_coords, polygons)

    # Classify center card
    center_cards = player_cards_dict["Center"]
    if center_cards:
        center_segmented = segmented_cards(center_cards[0]['crop'], plot=False)
        if center_segmented:
            center_value, center_colour, _ = classify_card(center_segmented[0])
            if center_value is not None:
                center_card = center_value
        else:
            center_card = "EMPTY"

    print(f"Center card: {center_card}")

    # Classify player cards
    player_names = ["Player 1", "Player 2", "Player 3", "Player 4"]
    for i, player_name in enumerate(player_names):
        cards = player_cards_dict[player_name]
        classified_cards = []

        for card_info in cards:
            card_crop = card_info['crop']
            card_value, card_colour, _ = classify_card(card_crop)

            if card_value is not None:
                classified_cards.append(card_value)

        # if cards were detected, join them with ';' and update player_cards
        if classified_cards:
            player_cards[i] = ";".join(classified_cards)

    for i, cards in enumerate(player_cards, start=1):
        print(f"Player {i} cards: {cards}")

    # Save results
    submission_df.at[index, "center_card"] = center_card
    submission_df.at[index, "active_player"] = active_player
    for i in range(4):
        submission_df.at[index, f"player_{i+1}_cards"] = player_cards[i]

# Save the submission file with a timestamp
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
submission_df.to_csv(f"data/output/{mode}_submission_{datetime_str}.csv", sep=",", index=False)
