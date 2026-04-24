import pandas as pd

submission_df = pd.read_csv("data/sample_submission.csv")
print(submission_df.head())

# remove ID 'L1000867' because it is not in the test set
submission_df = submission_df[submission_df["image_id"] != "L1000867"]
print(len(submission_df))
