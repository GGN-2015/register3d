import rotate_volume
from typing import Tuple
import cupy_fft_match
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation
import math
from tqdm import tqdm

# Find the position where the minimum value occurs in a 3D array
# Returns posX, posY, posZ, score
# where score is the minimum value of the original array
def find_3d_min_coords(arr_3d: cp.ndarray) -> Tuple[int, int, int, float]:
    min_val = cp.min(arr_3d)
    all_min_coords = cp.asnumpy(cp.argwhere(arr_3d == min_val))
    first_min_coords = [int(term) for term in all_min_coords[0].tolist()]
    assert len(first_min_coords) == 3
    return tuple(first_min_coords + [float(min_val)]) # type: ignore

# Check if there are values >= 0.5 around each position
def check_3d_neighborhood(arr_3d, threshold=0.5) -> cp.ndarray:
    binary_arr = (arr_3d >= threshold).astype(cp.uint8)
    kernel = cp.ones((5, 5, 5), dtype=cp.uint8)
    dilated_arr = binary_dilation(
        input=binary_arr,
        structure=kernel
    ).astype(cp.bool_)
    return dilated_arr

# Calculate weight matrix
def get_weight_array(image_part: cp.ndarray) -> cp.ndarray:
    ans = cp.zeros_like(image_part, dtype=image_part.dtype)
    image_geq_half = image_part >= 0.5
    image_lt_half  = cp.ones_like(image_part, dtype=cp.bool_)
    # Boundary weight is 0.5, weight for other positions is 1.0
    ans[image_lt_half & check_3d_neighborhood(image_geq_half)] = 0.5
    ans[image_geq_half] = 1.0
    return ans

# Direct matching without rotation
# Returns the optimal rotation result
def match_3d_data(full_image: cp.ndarray, image_part: cp.ndarray) -> Tuple[cp.ndarray, float]:
    match_array = cupy_fft_match.match_arr(full_image, image_part, weight_arr:=get_weight_array(image_part))

    # Find the position and value of the minimum value
    posX, posY, posZ, score = find_3d_min_coords(match_array)

    # Construct translated data information
    moved_image_part = cp.zeros_like(full_image, dtype=full_image.dtype)
    moved_image_part[
        posX:posX+image_part.shape[0],
        posY:posY+image_part.shape[1],
        posZ:posZ+image_part.shape[2]
    ] = image_part
    return moved_image_part, score / weight_arr.sum()

# Find the optimal rotation solution considering rotation
# The first parameter is the translated image_part with the same size as full_image
# The second parameter is the score of all valid pixels, and the third is the optimal rotation matrix
def match_3d_data_rotate(full_image: cp.ndarray, image_part: cp.ndarray) -> Tuple[cp.ndarray, float, cp.ndarray]:
    best_moved_image_part = None
    best_score = math.inf
    best_rotation_matrix = None
    for x_angle_raw in tqdm(range(-100, 100, 15)):
        x_angle = x_angle_raw / 10

        for y_angle_raw in range(-50, 50, 15):
            y_angle = y_angle_raw / 10

            for z_angle_raw in range(-50, 50, 15):
                z_angle = z_angle_raw / 10

                rotate_x = rotate_volume.create_rotation_matrix("x", x_angle)
                rotate_y = rotate_volume.create_rotation_matrix("y", y_angle)
                rotate_z = rotate_volume.create_rotation_matrix("z", z_angle)
                rotate_matrix = rotate_z @ rotate_y @ rotate_x
                rotated_volume = rotate_volume.rotate_volume(image_part, rotate_matrix)
                
                # If the size exceeds the limit after rotation, it is definitely not a good solution
                try:
                    moved_image_part, score = match_3d_data(full_image, rotated_volume)
                    if score < best_score:
                        best_moved_image_part, best_score, best_rotation_matrix = moved_image_part, score, rotate_matrix
                except:
                    pass

    return best_moved_image_part, best_score, best_rotation_matrix
