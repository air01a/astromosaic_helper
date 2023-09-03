import numpy as np
import cv2
from astropy.io import fits
import io
from scipy.signal import convolve2d

I16_BITS_MAX_VALUE=65535
HOT_PIXEL_RATIO=2

class Stretch:

    def __init__(self, target_bkg=0.25, shadows_clip=-2):
        self.shadows_clip = shadows_clip
        self.target_bkg = target_bkg

 
    def _get_avg_dev(self, data):
        """Return the average deviation from the median.

        Args:
            data (np.array): array of floats, presumably the image data
        """
        median = np.median(data)
        n = data.size
        median_deviation = lambda x: abs(x - median)
        avg_dev = np.sum( median_deviation(data) / n )
        return avg_dev

    def _mtf(self, m, x):
        """Midtones Transfer Function

        MTF(m, x) = {
            0                for x == 0,
            1/2              for x == m,
            1                for x == 1,

            (m - 1)x
            --------------   otherwise.
            (2m - 1)x - m
        }

        See the section "Midtones Balance" from
        https://pixinsight.com/doc/tools/HistogramTransformation/HistogramTransformation.html

        Args:
            m (float): midtones balance parameter
                       a value below 0.5 darkens the midtones
                       a value above 0.5 lightens the midtones
            x (np.array): the data that we want to copy and transform.
        """
        shape = x.shape
        x = x.flatten()
        zeros = x==0
        halfs = x==m
        ones = x==1
        others = np.logical_xor((x==x), (zeros + halfs + ones))

        x[zeros] = 0
        x[halfs] = 0.5
        x[ones] = 1
        x[others] = (m - 1) * x[others] / ((((2 * m) - 1) * x[others]) - m)
        return x.reshape(shape)

    def _get_stretch_parameters(self, data):
        """ Get the stretch parameters automatically.
        m (float) is the midtones balance
        c0 (float) is the shadows clipping point
        c1 (float) is the highlights clipping point
        """
        median = np.median(data)
        avg_dev = self._get_avg_dev(data)

        c0 = np.clip(median + (self.shadows_clip * avg_dev), 0, 1)
        m = self._mtf(self.target_bkg, median - c0)

        return {
            "c0": c0,
            "c1": 1,
            "m": m
        }

    def stretch(self, data):
        """ Stretch the image.

        Args:
            data (np.array): the original image data array.

        Returns:
            np.array: the stretched image data
        """

        # Normalize the data
        d = data / np.max(data)

        # Obtain the stretch parameters
        stretch_params = self._get_stretch_parameters(d)
        m = stretch_params["m"]
        c0 = stretch_params["c0"]
        c1 = stretch_params["c1"]

        # Selectors for pixels that lie below or above the shadows clipping point
        below = d < c0
        above = d >= c0

        # Clip everything below the shadows clipping point
        d[below] = 0

        # For the rest of the pixels: apply the midtones transfer function
        d[above] = self._mtf(m, (d[above] - c0)/(1 - c0))
        return d


class Image:

    UNDEF_EXP_TIME = -1
    """
    Represents an image, our basic processing object.
    Image data is a numpy array. Array's data type is unspecified for now
    but we'd surely benefit from enforcing one (float32 for example) as it will
    ease the development of any later processing code
    We also store the bayer pattern the image was shot with, if applicable.
    If image is from a sensor without a bayer array, the bayer pattern must be None.
    """

    def __init__(self, data):
        """
        Constructs an Image
        :param data: the image data
        :type data: numpy.ndarray
        """
        self._data = data
        self._bayer_pattern: str = ""
        self._origin: str = "UNDEFINED"
        self._destination: str = "UNDEFINED"
        self._ticket = ""
        self._exposure_time: float = Image.UNDEF_EXP_TIME

    def clone(self, keep_ref_to_data=False):
        """
        Clone an image
        :param keep_ref_to_data: don't copy numpy data. This allows light image clone
        :type keep_ref_to_data: bool
        :return: an image with global copied data
        :rtype: Image
        """
        new_image_data = self.data if keep_ref_to_data else self.data.copy()
        new_image = Image(new_image_data)
        new_image.bayer_pattern = self.bayer_pattern
        new_image.origin = self.origin
        new_image.destination = self.destination
        new_image.ticket = self.ticket
        new_image.exposure_time = self.exposure_time
        return new_image

    @property
    def exposure_time(self):
        return self._exposure_time

    @exposure_time.setter
    def exposure_time(self, value):
        self._exposure_time = value

    @property
    def destination(self):
        """
        Retrieves image destination
        :return: the destination
        :rtype: str
        """
        return self._destination

    @destination.setter
    def destination(self, destination):
        """
        Sets image destination
        :param destination: the image destination
        :type destination: str
        """
        self._destination = destination

    @property
    def ticket(self):
        """
        Retrieves image ticket
        :return: the ticket
        :rtype: str
        """
        return self._ticket

    @ticket.setter
    def ticket(self, ticket):
        """
        Sets image ticket
        :param ticket: the image ticket
        :type ticket: str
        """
        self._ticket = ticket

    @property
    def data(self):
        """
        Retrieves image data
        :return: image data
        :rtype: numpy.ndarray
        """
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def origin(self):
        """
        retrieves info on image origin.
        If Image has been read from a disk file, origin contains the file path
        :return: origin representation
        :rtype: str
        """
        return self._origin

    @origin.setter
    def origin(self, origin):
        self._origin = origin

    @property
    def bayer_pattern(self):
        """
        Retrieves the bayer pattern the image was shot with, if applicable.
        :return: the bayer pattern or None
        :rtype: str
        """
        return self._bayer_pattern

    @property
    def dimensions(self):
        """
        Retrieves image dimensions as a tuple.
        This is basically the underlying array's shape tuple, minus the color axis if image is color
        :return: the image dimensions
        :rtype: tuple
        """
        if self._data.ndim == 2:
            return self._data.shape

        dimensions = list(self.data.shape)
        dimensions.remove(min(dimensions))
        return dimensions

    @property
    def width(self):
        """
        Retrieves image width
        :return: image width in pixels
        :rtype: int
        """
        return max(self.dimensions)

    @property
    def height(self):
        """
        Retrieves image height
        :return: image height in pixels
        :rtype: int
        """
        return min(self.dimensions)

    @bayer_pattern.setter
    def bayer_pattern(self, bayer_pattern):
        self._bayer_pattern = bayer_pattern

    def needs_debayering(self):
        """
        Tells if image needs debayering
        :return: True if a bayer pattern is known and data does not have 3 dimensions
        """
        return self._bayer_pattern != "" and self.data.ndim < 3

    def is_color(self):
        """
        Tells if the image has color information
        image has color information if its data array has more than 2 dimensions
        :return: True if the image has color information, False otherwise
        :rtype: bool
        """
        return self._data.ndim > 2

    def is_bw(self):
        """
        Tells if image is black and white
        :return: True if no color info is stored in data array, False otherwise
        :rtype: bool
        """
        return self._data.ndim == 2 and self._bayer_pattern == ""

    def is_same_shape_as(self, other):
        """
        Is this image's shape equal to another's ?
        :param other: other image to compare shape with
        :type other: Image
        :return: True if shapes are equal, False otherwise
        :rtype: bool
        """
        return self._data.shape == other.data.shape

    def set_color_axis_as(self, wanted_axis):
        """
        Reorganise internal data array so color information is on a specified axis
        :param wanted_axis: The 0-based number of axis we want color info to be
        Image data is modified in place
        """

        if self._data.ndim > 2:

            # find what axis are the colors on.
            # axis 0-based index is the index of the smallest data.shape item
            shape = self._data.shape
            color_axis = shape.index(min(shape))

            if color_axis != wanted_axis:
                self._data = np.moveaxis(self._data, color_axis, wanted_axis)

    def __repr__(self):
        representation = (f'{self.__class__.__name__}('
                          f'ID={self.__hash__()}, '
                          f'Color={self.is_color()}, '
                          f'Exp. t={self.exposure_time}, '
                          f'Needs Debayer={self.needs_debayering()}, '
                          f'Bayer Pattern={self.bayer_pattern}, '
                          f'Width={self.width}, '
                          f'Height={self.height}, '
                          f'Data shape={self._data.shape}, '
                          f'Data type={self._data.dtype.name}, '
                          f'Origin={self.origin}, '
                          f'Destination={self.destination}')

        return representation
    
def debayer(image: Image):

    if image==None:
        return
    preferred_bayer_pattern = "AUTO"

    if preferred_bayer_pattern == "AUTO" and not image.needs_debayering():
        return

    cv2_debayer_dict = {

        "BG": cv2.COLOR_BAYER_BG2RGB,
        "GB": cv2.COLOR_BAYER_GB2RGB,
        "RG": cv2.COLOR_BAYER_RG2RGB,
        "GR": cv2.COLOR_BAYER_GR2RGB
    }

    if preferred_bayer_pattern != 'AUTO':
        bayer_pattern = preferred_bayer_pattern

        if image.needs_debayering() and bayer_pattern != image.bayer_pattern:
           print("The bayer pattern defined in your preferences differs from the one present in current image.")


    else:
        bayer_pattern = image.bayer_pattern

    cv_debay = bayer_pattern[3] + bayer_pattern[2]

    try:
        debayered_data = cv2.cvtColor(image.data, cv2_debayer_dict[cv_debay])
    except KeyError:
        print(f"unsupported bayer pattern : {bayer_pattern}")
    except cv2.error as error:
        print(f"Debayering error : {str(error)}")

    image.data = debayered_data

def _neighbors_average(data):
    """
    returns an array containing the means of all original array's pixels' neighbors
    :param data: the image to compute means for
    :return: an array containing the means of all original array's pixels' neighbors
    :rtype: numpy.Array
    """

    kernel = np.ones((3, 3))
    kernel[1, 1] = 0

    neighbor_sum = convolve2d(data, kernel, mode='same', boundary='fill', fillvalue=0)
    num_neighbor = convolve2d(np.ones(data.shape), kernel, mode='same', boundary='fill', fillvalue=0)

    return (neighbor_sum / num_neighbor).astype(data.dtype)

def hot_pixel_remover(image: Image):

    # the idea is to check every pixel value against its 8 neighbors
    # if its value is more than _HOT_RATIO times the mean of its neighbors' values
    # me replace its value with that mean

    # this can only work on B&W or non-debayered color images

    if not image:
        return None

    if not image.is_color():
        means = _neighbors_average(image.data)
        try:
            image.data = np.where(image.data / (means+0.00001) > HOT_PIXEL_RATIO, means, image.data)
        except Exception as exc:
            print("Error during hotpixelremover :( %s"%str(exc))
    else:
        print("Hot Pixel Remover cannot work on debayered color images.")


def open_fits(filename):
    try:

        with fits.open(filename) as fit:
            # pylint: disable=E1101
            data = fit[0].data
            header = fit[0].header

        image = Image(data)
        if 'BAYERPAT' in header:
            image.bayer_pattern = header['BAYERPAT']

        if 'EXPTIME' in header:
            image.exposure_time = header['EXPTIME']

        #hot_pixel_remover(image)
        #debayer(image)
        #if image.is_color():
        #        image.set_color_axis_as(0)
        
        return image
    except Exception as e:
        print(e)

def adapt(image):
    if image.is_color():
        image.set_color_axis_as(0)
    image.data = np.float32((image.data))

def normalize(image):
    if image.is_color():
        image.set_color_axis_as(2)
    image.data = np.uint16(np.clip(image.data, 0, I16_BITS_MAX_VALUE))
    return image

def stretch(image : Image, strength : float, algo: int =0):
    #n=1
    if (algo==0): # algo stretch with clipping (strength 0:1, default = 0.1)
        # Best for stars imaging
        if (image.is_color()):
            for i in range(0,image.data.shape[0]):
                min_val = np.percentile(image.data[i], strength)
                max_val = np.percentile(image.data[i], 100 - strength)
                image.data[i] = np.clip((image.data[i] - min_val) * (65535.0 / (max_val - min_val)), 0, 65535)
        else:
                min_val = np.percentile(image.data, strength)
                max_val = np.percentile(image.data, 100 - strength)
                image.data = np.clip((image.data - min_val) * (65535.0 / (max_val - min_val)), 0, 65535)
    elif (algo==1):
        # strength float : 0-1
        # Pixinsight MTF algorithm, best with nebula
        image.data = np.interp(image.data,
                                    (image.data.min(), image.data.max()),
                                    (0, I16_BITS_MAX_VALUE))
        if image.is_color():
            for channel in range(3):
                image.data[channel] = Stretch(target_bkg=strength).stretch(image.data[channel])
            else:
                image.data = Stretch(target_bkg=strength).stretch(image.data)
        image.data *= I16_BITS_MAX_VALUE

    elif (algo==2):
        # stddev method
        # strength between 0-8
        mean = np.mean(image.data)
        stddev = np.std(image.data)

        # Soustraire la moyenne et diviser par l'écart-type multiplié par le facteur de contraste
        contrast_factor = 1/(2000*strength)
        stretched_image = (image.data - mean) / (stddev * contrast_factor)

        # Tronquer les valeurs des pixels en dessous de zéro à zéro et au-dessus de 255 à 255
        stretched_image = np.clip(stretched_image, 0, 65535)

        # Convertir les valeurs des pixels en entiers
        image.data = stretched_image.astype(np.uint16)


def open_and_stretch_fits(path):
    image = open_fits(path)
    hot_pixel_remover(image)
    debayer(image)
    adapt(image)
    stretch(image, 0.2,0)
    normalize(image)
    return image




