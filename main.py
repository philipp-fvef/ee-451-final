import kagglehub

# Download latest version
path = kagglehub.competition_download("iapr-26-uno-vision-challenge")

print("Path to competition files:", path)
