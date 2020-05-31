import numpy as np

def rbg_to_gray(rgb):
    """Convert an RGB image to a grayscale one"""
    assert len(rgb)==3, "Invalid rgb array passed"
    return 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1140 * rgb[2]

def dist(p1, p2, lnorm=1):
    """Returns the distance as measured by the lnorm provided.
    Defaults to lnorm=1 (Manhattan distance).
    """
    return np.linalg.norm(np.array(p1)-np.array(p2), lnorm)