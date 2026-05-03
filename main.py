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

    # Classify the card
    center_card = "EMPTY"
    active_player = "EMPTY"
    player_cards = ["EMPTY"] * 4

    # Save results
    submission_df.at[index, "center_card"] = center_card
    submission_df.at[index, "active_player"] = active_player
    for i in range(4):
        submission_df.at[index, f"player_{i+1}_cards"] = player_cards[i]

# Save the submission file with a timestamp
datetime_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
submission_df.to_csv(f"data/submission_{datetime_str}.csv", index=False)
