import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img


def load_and_resample(cam_path, roi_path):
    """
    将 Grad-CAM 与 ROI 对齐到 MNI152 标准空间（以 CAM 为参考图像）
    """
    cam_img = nib.load(cam_path)
    roi_img = nib.load(roi_path)

    # 将 ROI 重采样到 CAM 的空间，保持与论文中对齐 MNI 的处理一致
    roi_resampled = resample_to_img(roi_img, cam_img, interpolation='nearest')

    return cam_img.get_fdata(), roi_resampled.get_fdata()


def compute_roi_metrics(cam_3d, roi_3d, thr=0.6):
    """
    计算 A_roi, M_roi, C_roi
    cam_3d: Grad-CAM 体数据 (3D)
    roi_3d: ROI mask（0/1）(3D)
    """

    # ROI 内的 CAM 值
    cam_roi_vals = cam_3d[roi_3d > 0]

    # A_roi: 平均激活强度
    A_roi = cam_roi_vals.mean()

    # M_roi: 最大激活强度
    M_roi = cam_roi_vals.max()

    # Grad-CAM 二值化
    cam_bin = (cam_3d > thr).astype(np.int32)

    # C_roi: 覆盖率 = CAM_bin ∩ ROI / ROI 体素数
    C_roi = cam_bin[roi_3d > 0].mean()

    return A_roi, M_roi, C_roi


def compute_multiple_rois(cam_path, roi_paths, thr=0.6):
    """
    对多个 ROI（海马、海马旁回、前楔叶、颞区）依次计算指标
    roi_paths: dict, 如 {'hipp': 'hipp.nii', 'phg': 'phg.nii', ...}
    """
    results = {}

    for roi_name, roi_path in roi_paths.items():
        cam_data, roi_data = load_and_resample(cam_path, roi_path)

        A_roi, M_roi, C_roi = compute_roi_metrics(cam_data, roi_data, thr)
        results[roi_name] = (A_roi, M_roi, C_roi)

    return results


roi_paths = {
    "hippocampus": "roi_hipp.nii.gz",
    "parahippocampal": "roi_phg.nii.gz",
    "precuneus": "roi_precuneus.nii.gz",
    "temporal": "roi_temporal.nii.gz"
}

cam_file = "subject01_cam.nii.gz"

results = compute_multiple_rois(cam_file, roi_paths, thr=0.6)

for roi, vals in results.items():
    print(f"{roi}: A_roi={vals[0]:.3f}, M_roi={vals[1]:.3f}, C_roi={vals[2]:.3f}")
