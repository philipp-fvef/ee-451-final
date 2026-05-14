"""
Image cropping and segmentation utilities for dividing playing cards into sectors.
"""

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


# Sector boundary coordinates
X_LEFT = 980
X_MID_LEFT = 1250
X_MID_RIGHT = 2700
X_RIGHT = 3100

Y_TOP_LEFT = 500
Y_TOP = 850
Y_BOTTOM = 1900
Y_BOTTOM_RIGHT = 2200


def show_sectors(img_path):
    """
    Display an image with sector boundaries drawn on it.
    
    Args:
        img_path: Path to the image file
    """
    img = Image.open(img_path)
    width, height = img.size

    plt.figure(figsize=(12, 8))
    plt.imshow(img)

    # Top-left step (horizontal + small vertical drop)
    plt.plot([0, X_LEFT], [Y_TOP_LEFT, Y_TOP_LEFT], color="black", linewidth=4)
    plt.plot([X_LEFT, X_LEFT], [Y_TOP_LEFT, Y_TOP], color="black", linewidth=4)

    # Main top horizontal line
    plt.plot([X_LEFT, X_RIGHT], [Y_TOP, Y_TOP], color="black", linewidth=4)

    # Left vertical (ONLY bottom segment, removed upper part)
    plt.plot([X_LEFT, X_LEFT], [Y_BOTTOM, height], color="black", linewidth=4)

    # Central vertical separators (define middle sectors)
    plt.plot([X_MID_LEFT, X_MID_LEFT], [Y_TOP, Y_BOTTOM], color="black", linewidth=4)
    plt.plot([X_MID_RIGHT, X_MID_RIGHT], [Y_TOP, Y_BOTTOM], color="black", linewidth=4)

    # Main bottom horizontal line
    plt.plot([X_LEFT, X_RIGHT], [Y_BOTTOM, Y_BOTTOM], color="black", linewidth=4)

    # Right vertical (ONLY upper segment, removed middle part)
    plt.plot([X_RIGHT, X_RIGHT], [0, Y_TOP], color="black", linewidth=4)

    # Bottom-right step (small vertical + horizontal extension)
    plt.plot([X_RIGHT, X_RIGHT], [Y_BOTTOM, Y_BOTTOM_RIGHT], color="black", linewidth=4)
    plt.plot([X_RIGHT, width], [Y_BOTTOM_RIGHT, Y_BOTTOM_RIGHT], color="black", linewidth=4)

    plt.axis("off")
    plt.show()


def get_sector_polygons(img):
    """
    Get polygon coordinates for each sector of the image.
    
    Args:
        img: PIL Image object
        
    Returns:
        dict: Dictionary mapping sector names to polygon coordinate lists
    """
    width, height = img.size

    polygons = {
        "Player 3": [
            (0, 0),
            (X_RIGHT, 0),
            (X_RIGHT, Y_TOP),
            (X_LEFT, Y_TOP),
            (X_LEFT, Y_TOP_LEFT),
            (0, Y_TOP_LEFT)
        ],

        "Player 4": [
            (0, Y_TOP_LEFT),
            (X_LEFT, Y_TOP_LEFT),
            (X_LEFT, Y_TOP),
            (X_MID_LEFT, Y_TOP),
            (X_MID_LEFT, Y_BOTTOM),
            (X_LEFT, Y_BOTTOM),
            (X_LEFT, height),
            (0, height)
        ],

        "Center": [
            (X_MID_LEFT, Y_TOP),
            (X_MID_RIGHT, Y_TOP),
            (X_MID_RIGHT, Y_BOTTOM),
            (X_MID_LEFT, Y_BOTTOM)
        ],

        "Player 2": [
            (X_RIGHT, 0),
            (width, 0),
            (width, Y_BOTTOM_RIGHT),
            (X_RIGHT, Y_BOTTOM_RIGHT),
            (X_RIGHT, Y_BOTTOM),
            (X_MID_RIGHT, Y_BOTTOM),
            (X_MID_RIGHT, Y_TOP),
            (X_RIGHT, Y_TOP)
        ],

        "Player 1": [
            (X_LEFT, Y_BOTTOM),
            (X_RIGHT, Y_BOTTOM),
            (X_RIGHT, Y_BOTTOM_RIGHT),
            (width, Y_BOTTOM_RIGHT),
            (width, height),
            (X_LEFT, height)
        ]
    }

    return polygons


def extract_sector(img, polygon):
    """
    Extract a sector from an image using a polygon mask.

    Args:
        img: PIL Image object
        polygon: List of (x, y) coordinates defining the sector polygon

    Returns:
        tuple: (sector_crop, mask_crop) - cropped sector image and its mask
    """
    width, height = img.size

    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(polygon, fill=255)

    bbox = mask.getbbox()

    sector = Image.new("RGB", (width, height), "white")
    sector.paste(img, mask=mask)

    sector_crop = sector.crop(bbox)
    mask_crop = mask.crop(bbox)

    return sector_crop, mask_crop


def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using ray casting algorithm.

    Args:
        point: tuple (x, y)
        polygon: list of (x, y) tuples defining the polygon

    Returns:
        bool: True if point is inside polygon, False otherwise
    """
    x, y = point
    n = len(polygon)
    inside = False

    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def assign_cards_to_players(detected_cards, polygons):
    """
    Assign detected cards to players based on their center coordinates.

    Args:
        detected_cards: list of card dicts with 'center' key (x, y)
        polygons: dict of sector polygons from get_sector_polygons()

    Returns:
        dict: {player_name: [cards]} mapping player regions to detected cards
    """
    player_cards = {
        "Center": [],
        "Player 1": [],
        "Player 2": [],
        "Player 3": [],
        "Player 4": []
    }

    for card in detected_cards:
        center = card['center']

        for sector_name, polygon in polygons.items():
            if point_in_polygon(center, polygon):
                player_cards[sector_name].append(card)
                break

    return player_cards
