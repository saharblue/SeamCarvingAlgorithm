import numpy as np
from PIL import Image
from numba import jit
from tqdm import tqdm
from abc import abstractmethod, abstractstaticmethod
from os.path import basename
import functools

    
def NI_decor(fn):
    def wrap_fn(self, *args, **kwargs):
        try:
            fn(self, *args, **kwargs)
        except NotImplementedError as e:
            print(e)
    return wrap_fn


class SeamImage:
    def __init__(self, img_path, vis_seams=True):
        """ SeamImage initialization.

        Parameters:
            img_path (str): image local path
            method (str) (a or b): a for Hard Vertical and b for the known Seam Carving algorithm
            vis_seams (bool): if true, another version of the original image shall be store, and removed seams should be marked on it
        """
        #################
        # Do not change #
        #################
        self.path = img_path
        
        self.gs_weights = np.array([[0.299, 0.587, 0.114]]).T
        
        self.rgb = self.load_image(img_path)
        self.resized_rgb = self.rgb.copy()

        self.vis_seams = vis_seams
        if vis_seams:
            self.seams_rgb = self.rgb.copy()
        
        self.h, self.w = self.rgb.shape[:2]
        
        try:
            self.gs = self.rgb_to_grayscale(self.rgb)
            self.resized_gs = self.gs.copy()
            self.cumm_mask = np.ones_like(self.gs, dtype=bool)
        except NotImplementedError as e:
            print(e)

        try:
            self.E = self.calc_gradient_magnitude()
        except NotImplementedError as e:
            print(e)

        self.backtrack_mat = np.zeros_like(self.E)
        self.M = np.zeros_like(self.E)
        self.mask = np.zeros_like(self.M)
        self.is_horizontal = False
        #################

        # additional attributes you might find useful
        self.seam_history = []
        self.seam_balance = 0

        # This might serve you to keep tracking original pixel indices 
        self.idx_map_h, self.idx_map_v = np.meshgrid(range(self.w), range(self.h))

    def rgb_to_grayscale(self, np_img):
        """ Converts a np RGB image into grayscale (using self.gs_weights).
        Parameters
            np_img : ndarray (float32) of shape (h, w, 3) 
        Returns:
            grayscale image (float32) of shape (h, w, 1)

        Guidelines & hints:
            Use NumpyPy vectorized matrix multiplication for high performance.
            To prevent outlier values in the boundaries, we recommend to pad them with 0.5
        """
        grayscale_img = np_img @ self.gs_weights
        return grayscale_img.squeeze()

    # @NI_decor
    def calc_gradient_magnitude(self):
        """ Calculate gradient magnitude of a grayscale image

        Returns:
            A gradient magnitude image (float32) of shape (h, w)

        Guidelines & hints:
            - In order to calculate a gradient of a pixel, only its neighborhood is required.
            - keep in mind that values must be in range [0,1]
            - np.gradient or other off-the-shelf tools are NOT allowed, however feel free to compare yourself to them
        """
        gs_gradient_x = np.subtract(self.resized_gs, np.roll(self.resized_gs, -1, axis=1))
        gs_gradient_y = np.subtract(self.resized_gs, np.roll(self.resized_gs, -1, axis=0))
        gradient_magnitude = np.sqrt(np.add(np.square(gs_gradient_x), np.square(gs_gradient_y)))

        return gradient_magnitude

    def calc_M(self):
        """Calculates the matrix M discussed in lecture (with forward-looking cost)
        and fills the backtracking matrix with the direction of the minimum cost path.
        Returns:
            An energy matrix M (float32) of shape (h, w)

        Guidelines & hints:
            As taught, the energy is calculated from top to bottom.
            You might find the function 'np.roll' useful.
        """
        # Initialize M with zeros and backtrack with -5
        M = np.zeros_like(self.E)
        self.backtrack_mat = np.full_like(self.E, -5, dtype=np.int32)

        # The first row of M is the same as the first row of the energy matrix E
        M[0, :] = self.E[0, :]

        # Precompute the shifted versions of the energy matrix E
        resized_gs_2d = self.resized_gs
        gs_shifted_right = np.roll(resized_gs_2d, 1, axis=1)
        gs_shifted_left = np.roll(resized_gs_2d, -1, axis=1)
        gs_shifted_down = np.roll(resized_gs_2d, 1, axis=0)

        # Compute the forward energy costs for the entire image
        c_left = np.abs(gs_shifted_right - gs_shifted_left) + np.abs(gs_shifted_down - gs_shifted_right)
        c_right = np.abs(gs_shifted_right - gs_shifted_left) + np.abs(gs_shifted_down - gs_shifted_left)
        c_up = np.abs(gs_shifted_right - gs_shifted_left)

        # Set the infinite costs for the first and last columns
        c_left[:, 0] = np.inf
        c_right[:, -1] = np.inf

        # Iterate over each row, starting from the second row
        for i in range(1, self.h):
            # Calculate the total costs for each direction
            cost_up = M[i - 1, :] + c_up[i, :]
            cost_left = np.roll(M[i - 1, :], 1) + c_left[i, :]
            cost_right = np.roll(M[i - 1, :], -1) + c_right[i, :]

            # Find the minimum forward energy path from the previous row
            total_costs = np.vstack((cost_left, cost_up, cost_right))
            min_costs_indices = np.argmin(total_costs, axis=0)
            M[i, :] = self.E[i, :] + np.choose(min_costs_indices, total_costs)

            # Update the backtracking matrix with the direction of the minimum energy path
            self.backtrack_mat[i, :] = min_costs_indices - 1  # -1 for left, 0 for up, 1 for right

        return M

    def seams_removal(self, num_remove):
        pass

    def seams_removal_horizontal(self, num_remove):
        pass

    def seams_removal_vertical(self, num_remove):
        pass

    def rotate_mats(self, clockwise=True):
        if clockwise:
            self.gs = np.rot90(self.gs, -1)
            self.rgb = np.rot90(self.rgb, -1)
            self.resized_gs = np.rot90(self.resized_gs, -1)
            self.resized_rgb = np.rot90(self.resized_rgb, -1)
            self.cumm_mask = np.rot90(self.cumm_mask, -1)
            self.seams_rgb = np.rot90(self.seams_rgb, -1)
            self.idx_map_h = np.rot90(self.idx_map_h, -1)
            self.idx_map_v = np.rot90(self.idx_map_v, 1)
            self.E = np.rot90(self.E, -1)
        else:
            self.gs = np.rot90(self.gs, 1)
            self.rgb = np.rot90(self.rgb, 1)
            self.resized_gs = np.rot90(self.resized_gs, 1)
            self.resized_rgb = np.rot90(self.resized_rgb, 1)
            self.cumm_mask = np.rot90(self.cumm_mask, 1)
            self.seams_rgb = np.rot90(self.seams_rgb, 1)
            self.idx_map_h = np.rot90(self.idx_map_h, 1)
            self.idx_map_v = np.rot90(self.idx_map_v, -1)
            self.E = np.rot90(self.E, 1)

        self.h, self.w = self.w, self.h
        self.idx_map_h, self.idx_map_v = self.idx_map_v, self.idx_map_h
        self.E = self.calc_gradient_magnitude()
        self.M = self.calc_M()

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.backtrack_mat = np.zeros_like(self.E, dtype=int).squeeze()
        self.M = self.calc_M()
        self.mask = np.ones_like(self.M, dtype=bool)

    def update_ref_mat(self):
        """
        Update the index map after removing a seam.
        For vertical seam removal, update the horizontal index map (self.idx_map_h).
        For horizontal seam removal, update the vertical index map (self.idx_map_v).

        Parameters:
        - is_vertical (bool): Flag indicating the orientation of the seam removal.
        """
        seam = self.seam_history[-1]

        if not self.is_horizontal:
            # Update for vertical seam removal
            for i, col in enumerate(seam):
                self.idx_map_h[i, col:] = np.roll(self.idx_map_h[i, col:], -1)
                # Since we're removing vertical seams, no need to update self.idx_map_v here
        else:
            # Update for horizontal seam removal
            for i, row in enumerate(seam):
                self.idx_map_v[row, i:] = np.roll(self.idx_map_v[row, i:], -1)
                # Since we're removing horizontal seams, no need to update self.idx_map_h here

    def remove_seam(self):
        pass

    def reinit(self):
        """ re-initiates instance
        """
        self.__init__(img_path=self.path)

    @staticmethod
    def load_image(img_path, format='RGB'):
        return np.asarray(Image.open(img_path).convert(format)).astype('float32') / 255.0


class VerticalSeamImage(SeamImage):
    def __init__(self, *args, **kwargs):
        """ VerticalSeamImage initialization.
        """
        super().__init__(*args, **kwargs)
        try:
            self.init_mats()
        except NotImplementedError as e:
            print(e)

    # @NI_decor
    def seams_removal(self, num_remove: int):
        """ Iterates num_remove times and removes num_remove vertical seams
        
        Parameters:
            num_remove (int): number of vertical seam to be removed

        Guidelines & hints:
        As taught, the energy is calculated from top to bottom.
        You might find the function np.roll useful.

        This step can be divided into a couple of steps:
            i) init/update matrices (E, M, backtracking matrix, saem mask) where:
                - E is the gradient magnitude matrix
                - M is the cost matrix
                - backtracking matrix is an idx matrix used to track the minimum seam from bottom up
                - mask is a boolean matrix for removed seams
            ii) fill in the backtrack matrix corresponding to M
            iii) seam backtracking: calculates the actual indices of the seam
            iv) index update: when a seam is removed, index mapping should be updated in order to keep track indices for next iterations
            v) seam removal: create the carved image with the reduced (and update seam visualization if desired)
            Note: the flow described below is a recommendation. You may implement seams_removal as you with, but it needs to supprt:
            - removing seams couple of times (call the function more than once)
            - visualize the original image with removed seams marked (for comparison)
        """

        for i in range(num_remove):
            self.init_mats()
            seam = self.backtrack_seam()
            self.seam_history.append(seam)
            for row_seam, col_seam in enumerate(seam):
                col_index = self.idx_map_h[row_seam, col_seam]
                row_index = self.idx_map_v[row_seam, col_seam]
                self.cumm_mask[row_index, col_index] = False

            self.update_ref_mat()
            self.remove_seam()

    def remove_seam(self):
        seam = self.seam_history[-1]  # Retrieve the last seam to be removed
        height, width, = self.resized_gs.shape
        new_width = width - 1  # New width after seam removal

        # Initialize new images with the updated width
        new_image_gs = np.zeros((height, new_width), dtype=self.resized_gs.dtype)
        new_image_rgb = np.zeros((height, new_width, 3), dtype=self.resized_rgb.dtype)  # Ensure 3 channels

        for i in range(height):
            j = seam[i]  # The column index of the seam for the current row
            # For the grayscale image, remove the seam pixel
            new_image_gs[i, :] = np.concatenate((self.resized_gs[i, :j], self.resized_gs[i, j + 1:]), axis=0)
            # For the RGB image, remove the seam pixel from each color channel
            new_image_rgb[i, :, :] = np.concatenate((self.resized_rgb[i, :j, :], self.resized_rgb[i, j + 1:, :]),
                                                    axis=0)
        # Update the images after seam removal
        self.resized_gs = new_image_gs
        self.resized_rgb = new_image_rgb

        self.h, self.w = self.resized_gs.shape

    def paint_seams(self):
        cumm_mask_rgb = np.stack([self.cumm_mask] * 3, axis=2)
        self.seams_rgb = np.where(cumm_mask_rgb, self.seams_rgb, [1,0,0])

        self.seam_history = []

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.backtrack_mat = np.zeros_like(self.E, dtype=int).squeeze()
        self.M = self.calc_M()
        self.mask = np.ones_like(self.M, dtype=bool)

    # @NI_decor
    def seams_removal_horizontal(self, num_remove):
        """ Removes num_remove horizontal seams

        Parameters:
            num_remove (int): number of horizontal seam to be removed
        """
        self.is_horizontal = True
        self.rotate_mats(clockwise=True)
        self.seams_removal(num_remove)
        self.paint_seams()
        self.rotate_mats(clockwise=False)

    # @NI_decor
    def seams_removal_vertical(self, num_remove):
        """ A wrapper for removing num_remove horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): umber of vertical seam to be removed
        """
        self.is_horizontal = False
        self.seams_removal(num_remove)
        self.paint_seams()

    # @NI_decor
    def backtrack_seam(self):
        """ Backtracks a seam for Seam Carving as taught in lecture
        """
        seam = []
        min_col = np.argmin(self.M[-1])
        seam.append(min_col)

        last_min_col = min_col
        for row in range(self.h - 2, 0, -1):
            index_in_backtrack_matrix = self.backtrack_mat[row + 1, last_min_col]
            last_min_col = int(last_min_col + index_in_backtrack_matrix)
            seam.append(last_min_col)

        seam.append(last_min_col + self.backtrack_mat[1, last_min_col])
        seam.reverse()
        return seam

    # @NI_decor
    def seams_addition(self, num_add: int):
        """ BONUS: adds num_add seamn to the image

            Parameters:
                num_add (int): number of horizontal seam to be removed

            Guidelines & hints:
            - This method should be similar to removal
            - You may use the wrapper functions below (to support both vertical and horizontal addition of seams)
            - Visualization: paint the added seams in green (0,255,0)

        """
        raise NotImplementedError("TODO: Implement SeamImage.seams_addition")
    
    # @NI_decor
    def seams_addition_horizontal(self, num_add):
        """ A wrapper for removing num_add horizontal seams (just a recommendation)

        Parameters:
            num_remove (int): number of horizontal seam to be added

        Guidelines & hints:
            You may find np.rot90 function useful

        """
        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_horizontal")

    # @NI_decor
    def seams_addition_vertical(self, num_add):
        """ A wrapper for removing num_add vertical seams (just a recommendation)

        Parameters:
            num_add (int): number of vertical seam to be added
        """

        raise NotImplementedError("TODO (Bonus): Implement SeamImage.seams_addition_vertical")

    # @NI_decor
    @staticmethod
    # @jit(nopython=True)
    def calc_bt_mat(M, E, backtrack_mat):
        """ Fills the BT back-tracking index matrix. This function is static in order to support Numba. To use it, uncomment the decorator above.
        
        Recommnded parameters (member of the class, to be filled):
            M: np.ndarray (float32) of shape (h,w)
            backtrack_mat: np.ndarray (int32) of shape (h,w): to be filled here

        Guidelines & hints:
            np.ndarray is a rederence type. changing it here may affected outsde.
        """
        raise NotImplementedError("TODO: Implement SeamImage.calc_bt_mat")
        h, w = M.shape


class SCWithObjRemoval(VerticalSeamImage):
    def __init__(self, active_masks=['Gemma'], *args, **kwargs):
        import glob
        """ VerticalSeamImage initialization.
        """
        self.obj_masks = {basename(img_path)[:-4]: self.load_image(img_path, format='L') for img_path in glob.glob('images/obj_masks/*')}
        self.active_masks = active_masks
        super().__init__(*args, **kwargs)

        try:
            self.preprocess_masks()
        except KeyError:
            print("TODO (Bonus): Create and add Jurassic's mask")
        
        try:
            self.M = self.calc_M()
        except NotImplementedError as e:
            print(e)

    def preprocess_masks(self):
        """ Mask preprocessing.
            different from images, binary masks are not continous. We have to make sure that every pixel is either 0 or 1.

            Guidelines & hints:
                - for every active mask we need make it binary: {0,1}
        """
        print('Active masks:', self.active_masks)
        self.obj_masks = {mask_name : mask for mask_name, mask in self.obj_masks.items() if mask_name in self.active_masks}
        for mask_name in self.obj_masks:
            mask = self.obj_masks[mask_name]
            binary_mask = np.where(mask > 0, 1, 0)
            self.obj_masks[mask_name] = binary_mask

    # @NI_decor
    def apply_mask(self):
        """ Applies all active masks on the image
            
            Guidelines & hints:
                - you need to apply the masks on other matrices!
                - think how to force seams to pass through a mask's object..
        """
        for mask_name in self.active_masks:
            print(self.obj_masks[mask_name].shape)

            mask = self.obj_masks[mask_name]
            self.E = np.where(mask == 1, 0, self.E + 1000)

    def init_mats(self):
        self.E = self.calc_gradient_magnitude()
        self.apply_mask()
        self.backtrack_mat = np.zeros_like(self.M, dtype=int)
        self.M = self.calc_M()
        self.mask = np.ones_like(self.M, dtype=bool)


    def reinit(self, active_masks):
        """ re-initiates instance
        """
        self.__init__(active_masks=active_masks, img_path=self.path)

    def remove_seam(self):
        """A wrapper for super().remove_seam method that also takes care of the masks."""
        super().remove_seam()  # Call the parent class's remove_seam method
        seam = self.seam_history[-1]  # Retrieve the last seam to be removed

        for mask_name in self.active_masks:
            print(self.obj_masks[mask_name].shape)

            mask = self.obj_masks[mask_name]
            new_mask = np.zeros((self.h, self.w), dtype=mask.dtype)  # Initialize a new mask with the updated width

            for i in range(self.h):
                j = seam[i]  # The column index of the seam for the current row
                # Keep all the pixels before the seam pixel, and all the pixels after the seam pixel in the current row
                new_mask[i, :] = np.concatenate((mask[i, :j], mask[i, j + 1:]), axis=0)

            self.obj_masks[mask_name] = new_mask  # Update the mask in the dictionary
            print(self.obj_masks[mask_name].shape)


def scale_to_shape(orig_shape: np.ndarray, scale_factors: list):
    """ Converts scale into shape

    Parameters:
        orig_shape (np.ndarray): original shape [y,x]
        scale_factors (list): scale factors for y,x respectively

    Returns
        the new shape
    """
    raise NotImplementedError("TODO: Implement scale_to_shape")


def resize_seam_carving(seam_img: SeamImage, shapes: np.ndarray):
    """ Resizes an image using Seam Carving algorithm

    Parameters:
        seam_img (SeamImage) The SeamImage instance to resize
        shapes (np.ndarray): desired shape (y,x)

    Returns
        the resized rgb image
    """
    raise NotImplementedError("TODO: Implement resize_seam_carving")


def bilinear(image, new_shape):
    """
    Resizes an image to new shape using bilinear interpolation method
    :param image: The original image
    :param new_shape: a (height, width) tuple which is the new shape
    :returns: the image resized to new_shape
    """
    in_height, in_width, _ = image.shape
    out_height, out_width = new_shape
    new_image = np.zeros(new_shape)
    ###Your code here###
    def get_scaled_param(org, size_in, size_out):
        scaled_org = (org * size_in) / size_out
        scaled_org = min(scaled_org, size_in - 1)
        return scaled_org
    scaled_x_grid = [get_scaled_param(x,in_width,out_width) for x in range(out_width)]
    scaled_y_grid = [get_scaled_param(y,in_height,out_height) for y in range(out_height)]
    x1s = np.array(scaled_x_grid, dtype=int)
    y1s = np.array(scaled_y_grid,dtype=int)
    x2s = np.array(scaled_x_grid, dtype=int) + 1
    x2s[x2s > in_width - 1] = in_width - 1
    y2s = np.array(scaled_y_grid,dtype=int) + 1
    y2s[y2s > in_height - 1] = in_height - 1
    dx = np.reshape(scaled_x_grid - x1s, (out_width, 1))
    dy = np.reshape(scaled_y_grid - y1s, (out_height, 1))
    c1 = np.reshape(image[y1s][:,x1s] * dx + (1 - dx) * image[y1s][:,x2s], (out_width, out_height, 3))
    c2 = np.reshape(image[y2s][:,x1s] * dx + (1 - dx) * image[y2s][:,x2s], (out_width, out_height, 3))
    new_image = np.reshape(c1 * dy + (1 - dy) * c2, (out_height, out_width, 3)).astype(int)
    return new_image


