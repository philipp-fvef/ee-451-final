# Notes and Ideas

## Dataset structure

We will always see:

* White background & Black rectangle
* Noisy background & Yellow circle

→ check for kind of image, then do if/else for the marker

## Image Processing

Don't do scaling as cards are always the (almost) exact same size

## Step-By-Step

0. categorise the image as white or noisy
1. find which player's turn it is
2. Chop up into Player 1-4 + centre parts
3. Segment out the cards
4. Feature Extraction
5. Classification: match the cards to the reference images and see which one works best
6. Write the return file and check
