import numpy as np


def chop_up_image_into_player_images(
    img: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Chop up the input image into 4 images for each player.
    Plus the center image of the table.

    Args:
    image: The input image

    Returns:
    A list of 4 images for each player and the center image
    """

    img_1 = 0
    img_2 = 0
    img_3 = 0
    img_4 = 0
    img_center = 0

    return [img_1, img_2, img_3, img_4], img_center


def is_image_noisy(img: np.ndarray) -> bool:
    """
    Is the image noisy or white background?

    Args:
    img: The input image

    Returns:
    1 if the image is noisy, 0 if the image is white

    """
    # Placeholder function - implement your noise detection logic here
    noisy = False
    return bool(noisy)


def find_player_turn(imgs: list[np.ndarray], noisy: bool = False) -> int:
    """
    Find the player whose turn it is.

    Args:
    imgs: A list of 4 input images for each player

    noisy: Whether the image is noisy or not

    Returns:
    i if it's player i's turn
    0 if it's not possible to determine the player's turn
    """

    assert len(imgs) == 4, "There should be exactly 4 images for the 4 players."

    player_turn = 0

    if noisy:
        player_turn = find_yellow_circle(imgs)
    else:
        player_turn = find_dark_rectangle(imgs)

    return player_turn


def find_dark_rectangle(imgs: list[np.ndarray]) -> int:
    """
    Find the player with the dark rectangle in their image.

    Args:
    imgs: A list of 4 input images for each player

    Returns:
    i if player i has a dark rectangle in their image
    0 if no player has a dark rectangle
    """

    return 0


def find_yellow_circle(imgs: list[np.ndarray]) -> int:
    """
    Find the player with the yellow circle in their image.

    Args:
    imgs: A list of 4 input images for each player

    Returns:
    i if player i has a yellow circle in their image
    0 if no player has a yellow circle
    """

    return 0


def segment_cards(img: np.ndarray) -> list[str]:
    """
    Segment the cards in the input image.

    Args:
    img: The input image

    Returns:
    A list of segmented card images
    """

    segmented_cards = []
    return segmented_cards
