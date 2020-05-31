from matplotlib import pyplot as plt
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

def make_image(pixel_array, filename=None):
    """Generate the image from the pixel array

    Args:
        pixel_array (np.array): pixel array to make into an image
        filename (str, optional): Where to save the image file. Defaults to None, which just shows the plot and does not save it.
    """
    
    plt.imshow(pixel_array, interpolation='nearest')
    plt.axis('off')
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename, bbox_inches='tight')

