import rotate_volume
from typing import Tuple
import cupy_fft_match
import cupy as cp
from cupyx.scipy.ndimage import binary_dilation
import math
from tqdm import tqdm

# 从一个 3d 数组中找最小值出现的位置
# 返回值为 posX, posY, posZ, score
# 其中 score 是原数组的最小值
def find_3d_min_coords(arr_3d: cp.ndarray) -> Tuple[int, int, int, float]:
    min_val = cp.min(arr_3d)
    all_min_coords = cp.asnumpy(cp.argwhere(arr_3d == min_val))
    first_min_coords = [int(term) for term in all_min_coords[0].tolist()]
    assert len(first_min_coords) == 3
    return tuple(first_min_coords + [float(min_val)]) # type: ignore

# 检查每一个位置周围是否有 >= 0.5 的值
def check_3d_neighborhood(arr_3d, threshold=0.5) -> cp.ndarray:
    binary_arr = (arr_3d >= threshold).astype(cp.uint8)
    kernel = cp.ones((5, 5, 5), dtype=cp.uint8)
    dilated_arr = binary_dilation(
        input=binary_arr,
        structure=kernel
    ).astype(cp.bool_)
    return dilated_arr

# 计算权重矩阵
def get_weight_array(image_part:cp.ndarray) -> cp.ndarray:
    ans = cp.zeros_like(image_part, dtype=image_part.dtype)
    image_geq_half = image_part >= 0.5
    image_lt_half  = cp.ones_like(image_part, dtype=cp.bool_)
    # 边界权重 0.5, 其他位置权重 1.0
    ans[image_lt_half & check_3d_neighborhood(image_geq_half)] = 0.5
    ans[image_geq_half] = 1.0
    return ans

# 不带旋转，直接匹配
# 返回最优的旋转情况
def match_3d_data(full_image:cp.ndarray, image_part:cp.ndarray) -> Tuple[cp.ndarray, float]:
    match_array = cupy_fft_match.match_arr(full_image, image_part, get_weight_array(image_part))

    # 找到其中最小值出现的位置以及最小值
    posX, posY, posZ, score = find_3d_min_coords(match_array)

    # 构建平移后的数据信息
    moved_image_part = cp.zeros_like(full_image, dtype=full_image.dtype)
    moved_image_part[
        posX:posX+image_part.shape[0],
        posY:posY+image_part.shape[1],
        posZ:posZ+image_part.shape[2]
    ] = image_part
    return moved_image_part, score

# 在考虑旋转的前提下找到最优的旋转解
def match_3d_data_rotate(full_image:cp.ndarray, image_part:cp.ndarray) -> Tuple[cp.ndarray, float]:
    best_moved_image_part = None
    best_score = math.inf
    for x_angle_raw in tqdm(range(-150, 150, 15)):
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
                
                # 如果旋转之后尺寸超过了界限，那肯定不是一个好的解
                try:
                    moved_image_part, score = match_3d_data(full_image, rotated_volume)
                    if score < best_score:
                        best_moved_image_part, best_score = moved_image_part, score
                except:
                    pass

    return best_moved_image_part, best_score

# 二值化
def binary_value(arr:cp.ndarray) -> cp.ndarray:
    arr[arr <  0.5] = 0.0
    arr[arr >= 0.5] = 1.0
    return arr

if __name__ == "__main__":
    import image2numpy
    import visualize_volume
    full_image = image2numpy.load_png_to_3d_array("/run/media/neko/Cold_001/Archive/9999-99-80_研究生工作相关/2025-11-25_Ct-to-bone/03-把骨骼掩码切片还原回体数据/2025-11-25_3D可视化骨组织/output_bone_img_path/")
    # image_part = image2numpy.load_png_to_3d_array("/run/media/neko/Cold_001/Archive/9999-99-80_研究生工作相关/2025-12-08_一些测试用的临时图片/tmpDir/")
    image_part = image2numpy.load_png_to_3d_array("/run/media/neko/Cold_001/Archive/9999-99-80_研究生工作相关/2025-11-25_MRI-to-bone/03-把骨骼掩码切片还原回体数据/2025-11-25_3D可视化骨组织/output_bone_img_path/")
    full_image = binary_value(cp.asarray(full_image))
    image_part = binary_value(cp.asarray(image_part))

    moved_image_part, score = match_3d_data_rotate(full_image, image_part)
    full_image = -full_image
    full_image[moved_image_part >= 0.5] = 1.0
    visualize_volume.visualize_volume(cp.asnumpy(full_image))
