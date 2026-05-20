import cv2
import os
import pandas as pd
from PIL import Image

from src.config import load_config, set_global_config
from src.init import initialize_reference_images
from src.cropping import get_sector_polygons, assign_cards_to_players
from src.active import detect_active_player
from src.segmentation import segmented_cards
from src.classify import classify_card
from src.metrics import calculate_metrics

MODE = "test"  # test or train

if MODE == "test":
    img_dir = "test_images"
elif MODE == "train":
    img_dir = "train_images"
else:
    raise ValueError("Invalid mode. Choose 'test' or 'train'.")

headers = [
    "image_id",
    "center_card",
    "active_player",
    "player_1_cards",
    "player_2_cards",
    "player_3_cards",
    "player_4_cards",
]

# Create the submission dataframe with the specified headers
submission_df = pd.DataFrame(columns=headers)

# Populate the image_id column with the IDs from the images in the specified directory
image_files = sorted(os.listdir(img_dir))
submission_df["image_id"] = [os.path.splitext(filename)[0] for filename in image_files]

# remove ID 'L1000867' because it is not in the test set
submission_df = submission_df[submission_df["image_id"] != "L1000867"]
print(submission_df.head())

# Load and set global configuration
config = load_config("config.json")
set_global_config(config)

# initialise reference images and compute features
initialize_reference_images()

# iterate over rows of the submission dataframe
for index, row in submission_df.iterrows():

    active_player = "EMPTY"
    center_card = "EMPTY"
    player_cards = ["EMPTY"] * 4

    image_id = row["image_id"]
    print(f"Processing {image_id}...")

    image_path = os.path.join(img_dir, f"{image_id}.jpg")

    # Load image
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil_rgb = Image.fromarray(img_rgb)

    active_player = detect_active_player(img_rgb)["active_player"]
    print(f"Active player: {active_player}")

    # Get sector polygons for player regions
    polygons = get_sector_polygons(img_pil_rgb)

    # Detect all cards in the full image with coordinates
    detected_cards_with_coords = segmented_cards(
        img_rgb, return_coords=True, plot=False
    )

    # Assign cards to players based on their coordinates
    player_cards_dict = assign_cards_to_players(detected_cards_with_coords, polygons)

    # Classify center card
    center_cards = player_cards_dict["Center"]
    if center_cards:
        center_segmented = segmented_cards(center_cards[0]["crop"], plot=False)
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
            card_crop = card_info["crop"]
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

# Save the submission file
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
submission_path = os.path.join(output_dir, f"{MODE}_submission.csv")
submission_df.to_csv(submission_path, sep=",", index=False)

if MODE == "train":
    calculate_metrics(submission_path, "train.csv")
