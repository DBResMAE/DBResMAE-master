import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.spatial.distance import directed_hausdorff, cdist
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Callable
from similaritymeasures import frechet_distance

def calculate_psnr(img1, img2, data_range: Optional[float] = None) -> float:
    """
    计算两张图像（或3D体素）的峰值信噪比 (PSNR)。
    参数:
        img1 协调前的图像
        img2 协调后的图像
        data_range (float, optional): 图像数据的动态范围 。
    返回:
        float: PSNR 值 (单位: dB)。
    """
    if img1.shape != img2.shape:
        raise ValueError("图像尺寸必须相同.")

    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')

    if data_range is None:
        data_range = np.max(img1) - np.min(img1)

    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


def calculate_ssim(img1, img2, data_range: Optional[float] = None) -> float:
    """
    计算两张图像（或3D体素）的结构相似性指数 (SSIM)。
    参数:
        img1 协调前的图像
        img2 协调后的图像
        data_range (float, optional): 图像数据的动态范围。
    返回:
        float: SSIM 值 (范围 [0, 1])。
    """
    if img1.shape != img2.shape:
        raise ValueError("图像尺寸必须相同.")


    if img1.ndim == 3:
        if data_range is None:
            max_val = max(np.max(img1), np.max(img2))
            min_val = min(np.min(img1), np.min(img2))
            data_range = max_val - min_val

        score, _ = ssim(img1, img2, data_range=data_range, win_size=None, full=True, channel_axis=None)
        return score
    else:
        return ssim(img1, img2, data_range=data_range, channel_axis=None)


def calculate_dice_coefficient(mask1, mask2) -> float:
    """
    计算两个二值掩模的 Dice 相似系数 (DSC)。
    参数:
        mask1 : 二值掩模1。
        mask2 : 二值掩模2。
    返回:
        float: Dice 系数值 (范围 [0, 1])。
    """
    if mask1.shape != mask2.shape:
        raise ValueError("掩模尺寸必须相同.")

    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # 交集 (Intersection)
    intersection = np.sum(mask1 & mask2)
    # 和 (Sum of elements)
    sum_of_masks = np.sum(mask1) + np.sum(mask2)

    if sum_of_masks == 0:
        return 1.0 if intersection == 0 else 0.0

    dsc = (2. * intersection) / sum_of_masks
    return dsc


def calculate_hd95(mask1, mask2) -> float:
    """
    计算两个二值掩模的 95th Percentile Hausdorff Distance (HD95)。
    参数:
        mask1  二值掩模1。
        mask2  二值掩模2。
    返回:
        float: HD95 值。
    """
    if mask1.shape != mask2.shape:
        raise ValueError("掩模尺寸必须相同.")

    # 提取点集 (体素坐标)
    points1 = np.argwhere(mask1 > 0)
    points2 = np.argwhere(mask2 > 0)

    if len(points1) == 0 or len(points2) == 0:
        return 0.0  # 至少一个掩模为空，HD95为 0

    # 计算欧氏距离矩阵
    distance_matrix = cdist(points1, points2, metric='euclidean')

    min_dist_1_to_2 = np.min(distance_matrix, axis=1)
    min_dist_2_to_1 = np.min(distance_matrix, axis=0)

    hd95_1_to_2 = np.percentile(min_dist_1_to_2, 95)
    hd95_2_to_1 = np.percentile(min_dist_2_to_1, 95)

    # 对称 HD95 是两个单侧距离中的最大值
    hd95 = max(hd95_1_to_2, hd95_2_to_1)

    return hd95


def calculate_frechet_distance(curve1_points, curve2_points) -> float:
    """
    计算两条曲线（点集）之间的离散 Fréchet 距离。
    参数:
        curve1_points: 曲线1 的坐标点集
        curve2_points: 曲线2 的坐标点集

    返回:
        float: Fréchet 距离值。
    """
    try:
        from similaritymeasures import frechet_distance
        return frechet_distance(curve1_points, curve2_points)
    except ImportError:
        print("警告: 缺少 'similaritymeasures' 库，无法精确计算 Fréchet 距离。")
        print("请运行 'pip install similaritymeasures'。返回一个简化版的距离。")

        # Fallback: 使用 Hausdorff 距离作为替代（不准确但接近）
        return max(directed_hausdorff(curve1_points, curve2_points)[0],
                   directed_hausdorff(curve2_points, curve1_points)[0])


def calculate_3d_frechet_distance(points_fixed, points_registered) -> float:
    """
    计算两个 3D 坐标点集之间的离散 Fréchet 距离
    参数:
        points_fixed 泄题案前
        points_registered : 协调后的图像
    返回:
        float: 离散 Fréchet 距离值。
    """

    if points_fixed.ndim != 2 or points_fixed.shape[1] != 3:
        raise ValueError(f"points_fixed 必须是 N x 3 的形状，当前形状为 {points_fixed.shape}")

    if points_registered.ndim != 2 or points_registered.shape[1] != 3:
        raise ValueError(f"points_registered 必须是 M x 3 的形状，当前形状为 {points_registered.shape}")


    try:
        fd_value = frechet_distance(points_fixed, points_registered)
        return fd_value
    except NotImplementedError:
        # 依赖库未安装时的错误处理
        return np.nan


results_PSNR = calculate_psnr(img_fixed, img_moving_registered, data_range=data_range)

results_SSIM = calculate_ssim(img_fixed, img_moving_registered, data_range=data_range)

results_DSC = calculate_dice_coefficient(mask_fixed, mask_moving_registered)

results_HD95 = calculate_hd95(mask_fixed, mask_moving_registered)

results_FD = calculate_3d_frechet_distance(key_points_fixed, key_points_moving_registered)


