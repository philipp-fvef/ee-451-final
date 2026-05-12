import pandas as pd
import numpy as np
import cv2
from PIL import Image

from utils.utils import *
from utils.lab_01_utils import *
from utils.lab_02_utils import *
from utils.lab_03_utils import *

from src.cropping import get_sector_polygons, extract_sector
from src.segmentation import segmented_cards
from src.classify import classify_card
from utils.config import load_config, set_global_config


submission_df = pd.read_csv("data/sample_submission.csv")
print(submission_df.head())

# remove ID 'L1000867' because it is not in the test set
submission_df = submission_df[submission_df["image_id"] != "L1000867"]
print(len(submission_df))

# Load and set global configuration
config = load_config("config.json")
set_global_config(config)

# iterate over rows of the submission dataframe
for index, row in submission_df.iterrows():
    image_id = row["image_id"]
    print(f"Processing {image_id}...")

    image_path = os.path.join("data/test_images", f"{image_id}.jpg")

    # Crop the image into player areas and the center card area
    img = Image.open(image_path).convert("RGB")
    polygons = get_sector_polygons(img)

    # Extract center card and convert to numpy array in BGR format (OpenCV format)
    center_card_pil, _ = extract_sector(img, polygons["Center"])
    center_card_img = cv2.cvtColor(np.array(center_card_pil), cv2.COLOR_RGB2BGR)

    # Extract player cards and convert to numpy arrays in BGR format
    player_imgs = []
    for player_name in ["Player 1", "Player 2", "Player 3", "Player 4"]:
        player_pil, _ = extract_sector(img, polygons[player_name])
        player_img_bgr = cv2.cvtColor(np.array(player_pil), cv2.COLOR_RGB2BGR)
        player_imgs.append(player_img_bgr)

    active_player = "EMPTY"
    center_card = "EMPTY"
    player_cards = ["EMPTY"] * 4

    # ----------------------
    # Classify the active player
    # p1, p2, p3, p4
    # ----------------------

    center_segmented = segmented_cards(center_card_img)
    if center_segmented:
        center_value, center_colour, _ = classify_card(center_segmented[0])
    else:
        center_value = "EMPTY"

    # Classify the player cards
    for i, player_img in enumerate(player_imgs):
        cropped_cards = []
        cards = []

        cropped_cards = segmented_cards(player_img)

        for cropped_card in cropped_cards:

            card_value, card_colour, _ = classify_card(cropped_card)
            cards.append(card_value)

        # if cards were detected, join them with ';' and update player_cards
        if cards: player_cards[i] = ";".join(cards)

    # Save results
    submission_df.at[index, "center_card"] = center_value
    submission_df.at[index, "active_player"] = active_player
    for i in range(4):
        submission_df.at[index, f"player_{i+1}_cards"] = player_cards[i]

# Save the submission file with a timestamp
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
submission_df.to_csv(f"data/submission_{datetime_str}.csv", sep=",", index=False)
