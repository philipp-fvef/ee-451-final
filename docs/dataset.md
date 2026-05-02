# Dataset Description

## Overview

The dataset consists of images of UNO game scenes together with structured annotations describing the game state.

Each image captures a tabletop setup where multiple players are holding cards. The goal is to extract the center card, the active player, and the cards held by each player.

The dataset is split into:

* Training set: images + ground truth annotations
* Test set: images only (labels hidden)

## Data Format

### Images

Each image contains:

* a center card on the table
* up to four players, each holding a set of cards

Scenes may vary in:

* number of players
* background
* card orientations

### Annotations (Training Set)

Each image corresponds to one row in a CSV file with the following structure: 
`image_id,center_card,active_player,player_1_cards,player_2_cards,player_3_cards,player_4_cards`

### Field Description

* image_id: unique identifier of the image
* center_card: card placed at the center of the table
* active_player: player whose turn it is (indicated with a token next to the player)
* player_k_cards (k = 1,2,3,4): set of cards held by each player

### Card Representation

Cards are represented as strings combining the first letter of the color and value/action. e.g. `r_5;b_skip;y_reverse;wild`

### Multiple Cards

Cards are separated by semicolons `;` order does not matter

### Duplicate Cards

If a player holds the same card multiple times, it must be explicitly repeated.

### Player Indexing

Player positions are fixed and must be respected:

* player_1 → bottom of the image
* layer_2 → right
* player_3 → top
* player_4 → left

Players are ordered counter-clockwise.

### Missing Players

* Not all images contain four players.
* Some player positions may be empty.
* Missing players can occur at any index.
* Their corresponding fields must be treated as having no cards

### Important Note on Empty Values

Due to Kaggle submission constraints:

* Empty fields are not allowed in submission files
* Instead, you must use the token: EMPTY
* This applies to players with no cards/missing players.