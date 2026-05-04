import pandas as pd

from utils.utils import *
from utils.lab_01_utils import *
from utils.lab_02_utils import *
from utils.lab_03_utils import *

submission_df = pd.read_csv("data/sample_submission.csv")
print(submission_df.head())

# remove ID 'L1000867' because it is not in the test set
submission_df = submission_df[submission_df["image_id"] != "L1000867"]
print(len(submission_df))

# iterate over rows of the submission dataframe
for index, row in submission_df.iterrows():
    image_id = row["image_id"]
    print(f"Processing {image_id}...")

    image_path = os.path.join("data/test_images", f"{image_id}.jpg")

    # Crop the image into player areas and the center card area
    center_card_img, player_imgs = None, [None] * 4

    active_player = "EMPTY"
    center_card = "EMPTY"
    player_cards = ["EMPTY"] * 4

    # ----------------------
    # Classify the active player
    # p1, p2, p3, p4
    # ----------------------

    # ----------------------
    # Classify the center card
    # ----------------------

    # Classify the player cards
    for i, player_img in enumerate(player_imgs):
        cropped_cards = []
        cards = []

        # ----------------------
        # segment out the cards
        # ----------------------

        for cropped_card in cropped_cards:

            # ----------------------
            # classify the card
            # ----------------------
            
            card_value = None
            cards.append(card_value)
        # if cards were detected, join them with ';' and update player_cards
        if cards: player_cards[i] = ";".join(cards)

    # Save results
    submission_df.at[index, "center_card"] = center_card
    submission_df.at[index, "active_player"] = active_player
    for i in range(4):
        submission_df.at[index, f"player_{i+1}_cards"] = player_cards[i]

# Save the submission file with a timestamp
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
submission_df.to_csv(f"data/submission_{datetime_str}.csv", sep=",", index=False)
