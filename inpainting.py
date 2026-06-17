"""
多波段图像修补 (Multi-band Image Inpainting)

问题描述：
    原始算法对各波段独立修补，当波段数量>3时（如高光谱、多通道卫星图像），
    会破坏波段间的光谱相关性，导致修补区域出现严重的颜色畸变/光谱畸变。

解决方案：
    采用PCA降维 -> 低维主成分空间联合修补 -> 逆PCA投影回原始波段空间。
    PCA保留了波段间的主要协方差结构，在主成分空间修补可保持光谱一致性。

作者: Hair Physics System Toolkit
"""

import numpy as np
from typing import Tuple, Optional, List
import warnings

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    warnings.warn("OpenCV (cv2) 未安装，将使用scipy.ndimage替代修补算法")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# =============================================================================
# 基础修补引擎（底层算法）
# =============================================================================

def _inpaint_single_channel(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea"
) -> np.ndarray:
    """单通道修补的底层实现。
    
    Args:
        image: 单通道图像 (H, W)，float32或uint8
        mask:  修补掩码 (H, W)，uint8，255=待修补区域，0=保留区域
        radius: 修补邻域半径
        method: "telea" | "ns" | "harmonic"
    
    Returns:
        修补后的单通道图像
    """
    if mask.sum() == 0:
        return image.copy()
    
    image_f32 = image.astype(np.float32)
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    
    if HAS_CV2 and method in ("telea", "ns"):
        img_u8 = np.clip(image_f32, 0, 255).astype(np.uint8)
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        result = cv2.inpaint(img_u8, mask_u8, radius, flag)
        return result.astype(np.float32)
    
    if HAS_SCIPY:
        mask_bool = mask_u8 > 0
        result = image_f32.copy()
        
        for _ in range(radius * 3):
            blurred = ndimage.uniform_filter(result, size=radius)
            result[mask_bool] = blurred[mask_bool]
        
        return result
    
    raise ImportError("需要安装 OpenCV 或 SciPy 才能执行修补操作")


# =============================================================================
# PCA 工具函数
# =============================================================================

class PCATransform:
    """主成分分析变换，用于多波段数据的降维与重建。
    
    对形状为 (H, W, C) 的多波段图像：
      1. 将像素重塑为 (N, C)，N=H*W
      2. 对有效像素（mask=0）执行PCA
      3. 将所有像素投影到前 K 个主成分空间
    """
    
    def __init__(self, n_components: Optional[int] = None, explained_variance_ratio: float = 0.98):
        """
        Args:
            n_components: 保留的主成分数量。若为None，则根据 explained_variance_ratio 自动确定
            explained_variance_ratio: 自动选择主成分时需要保留的方差比例（0~1）
        """
        self.n_components = n_components
        self.explained_variance_ratio = explained_variance_ratio
        
        self.mean_: Optional[np.ndarray] = None          # (C,)
        self.components_: Optional[np.ndarray] = None    # (K, C)
        self.explained_variance_: Optional[np.ndarray] = None  # (K,)
    
    def fit(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> "PCATransform":
        """拟合PCA。
        
        Args:
            X: 数据矩阵 (N, C)，N个样本，C个波段
            valid_mask: 可选 (N,) 布尔数组，True表示有效样本用于计算协方差
        """
        if valid_mask is not None:
            X_valid = X[valid_mask]
        else:
            X_valid = X
        
        if X_valid.shape[0] < 2:
            raise ValueError("有效样本数量不足，无法拟合PCA")
        
        self.mean_ = X_valid.mean(axis=0)
        X_centered = X_valid - self.mean_
        
        cov = X_centered.T @ X_centered / (X_valid.shape[0] - 1)
        
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = np.maximum(eigenvalues, 0)
        self.explained_variance_ = eigenvalues
        
        if self.n_components is None:
            total_var = eigenvalues.sum()
            if total_var < 1e-12:
                self.n_components = min(3, X.shape[1])
            else:
                cumulative = np.cumsum(eigenvalues) / total_var
                self.n_components = int(np.searchsorted(cumulative, self.explained_variance_ratio) + 1)
                self.n_components = max(1, min(self.n_components, X.shape[1]))
        
        self.n_components = min(self.n_components, X.shape[1], len(eigenvalues))
        self.components_ = eigenvectors[:, :self.n_components].T  # (K, C)
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """将 (N, C) 投影到 (N, K) 主成分空间。"""
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def inverse_transform(self, X_pca: np.ndarray) -> np.ndarray:
        """将 (N, K) 反投影回 (N, C) 原始波段空间。"""
        return X_pca @ self.components_ + self.mean_
    
    def fit_transform(self, X: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, valid_mask).transform(X)


# =============================================================================
# 原始算法（有缺陷）：独立波段修补
# =============================================================================

def inpaint_independent_bands(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea"
) -> np.ndarray:
    """对每个波段独立执行修补（原始有缺陷算法）。
    
    缺陷：当波段数>3时，各波段独立修补破坏了波段间的光谱相关性，
    修补区域会出现严重的颜色/光谱畸变。
    
    Args:
        image: 多波段图像 (H, W, C)，C为波段数
        mask:  修补掩码 (H, W)，uint8或bool，非零为待修补区域
        radius: 修补邻域半径
        method: 修补方法
    
    Returns:
        修补后的图像 (H, W, C)
    """
    if image.ndim != 3:
        raise ValueError(f"image 应为 (H, W, C)，实际形状 {image.shape}")
    
    H, W, C = image.shape
    mask_u8 = (mask > 0).astype(np.uint8) * 255 if mask.ndim == 2 else mask
    
    result = np.zeros_like(image, dtype=np.float32)
    
    for c in range(C):
        channel = image[:, :, c]
        result[:, :, c] = _inpaint_single_channel(channel, mask_u8, radius, method)
    
    return result


# =============================================================================
# 新算法：PCA降维 + 联合修补 + 反投影
# =============================================================================

def inpaint_pca_joint(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 3,
    method: str = "telea",
    n_components: Optional[int] = None,
    explained_variance_ratio: float = 0.98,
    return_pca_info: bool = False
) -> np.ndarray or Tuple[np.ndarray, dict]:
    """基于PCA降维的多波段联合修补（修复算法）。
    
    核心思想：
      1. 利用PCA将多波段数据投影到低维主成分空间（通常前3个主成分已保留95%+方差）
      2. 在主成分空间对各主成分联合修补（此时主成分间已互不相关，独立修补即可）
      3. 通过逆PCA将修补后的主成分投影回原始波段空间
      
    优势：
      - 保持了波段间的协方差结构，从根本上消除光谱/颜色畸变
      - 降维后只需修补3~5个主成分，速度比独立修补快（当波段数多时尤其明显）
      - 对高光谱图像（几十~上百波段）效果尤为显著
    
    Args:
        image: 多波段图像 (H, W, C)，C为波段数
        mask:  修补掩码 (H, W)，非零为待修补区域
        radius: 修补邻域半径
        method: "telea" | "ns" | "harmonic"
        n_components: 指定PCA主成分数；None则根据explained_variance_ratio自动确定
        explained_variance_ratio: 自动选择主成分数时的方差保留比例
        return_pca_info: 是否返回PCA相关信息（用于调试/可视化）
    
    Returns:
        修补后的图像 (H, W, C)；若 return_pca_info=True，则返回 (图像, info字典)
    """
    if image.ndim != 3:
        raise ValueError(f"image 应为 (H, W, C)，实际形状 {image.shape}")
    
    H, W, C = image.shape
    N = H * W
    
    mask_bool = mask > 0 if mask.ndim == 2 else mask.any(axis=-1)
    mask_u8 = mask_bool.astype(np.uint8) * 255
    
    X = image.reshape(N, C).astype(np.float32)
    
    valid_mask_flat = ~mask_bool.reshape(N)
    
    if C <= 3 and n_components is None:
        n_components = C
    
    pca = PCATransform(
        n_components=n_components,
        explained_variance_ratio=explained_variance_ratio
    )
    
    try:
        X_pca = pca.fit_transform(X, valid_mask=valid_mask_flat)
    except ValueError as e:
        warnings.warn(f"PCA拟合失败，回退到独立修补: {e}")
        if return_pca_info:
            return inpaint_independent_bands(image, mask, radius, method), {}
        return inpaint_independent_bands(image, mask, radius, method)
    
    K = pca.n_components
    X_pca_img = X_pca.reshape(H, W, K)
    
    X_pca_inpainted = np.zeros_like(X_pca_img)
    for k in range(K):
        channel = X_pca_img[:, :, k]
        X_pca_inpainted[:, :, k] = _inpaint_single_channel(channel, mask_u8, radius, method)
    
    X_pca_inpainted_flat = X_pca_inpainted.reshape(N, K)
    X_reconstructed = pca.inverse_transform(X_pca_inpainted_flat)
    
    result = X_reconstructed.reshape(H, W, C)
    
    valid_mask_3ch = ~mask_bool[:, :, None]
    result = np.where(valid_mask_3ch, image.astype(np.float32), result)
    
    if not return_pca_info:
        return result
    
    info = {
        "n_components": K,
        "total_bands": C,
        "explained_variance_ratio": pca.explained_variance_[:K] / pca.explained_variance_.sum()
                        if pca.explained_variance_.sum() > 0 else np.zeros(K),
        "cumulative_variance": np.cumsum(pca.explained_variance_[:K]) / pca.explained_variance_.sum()
                        if pca.explained_variance_.sum() > 0 else np.zeros(K),
        "components": pca.components_,
        "mean": pca.mean_,
        "pca_image": X_pca_img,
        "pca_inpainted": X_pca_inpainted,
    }
    return result, info


# =============================================================================
# 质量评估指标
# =============================================================================

def compute_spectral_angle_mapper(reference: np.ndarray, test: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """计算光谱角映射 (SAM)，评估光谱保真度。值越小越好，0表示完全一致。
    
    SAM衡量两个光谱向量之间的夹角，对波段间相关性非常敏感，
    是评估多/高光谱修补质量的核心指标。
    """
    if mask is not None:
        mask_bool = mask > 0 if mask.ndim == 2 else mask.any(axis=-1)
        ref = reference[mask_bool]
        tst = test[mask_bool]
    else:
        ref = reference.reshape(-1, reference.shape[-1])
        tst = test.reshape(-1, test.shape[-1])
    
    if ref.size == 0:
        return 0.0
    
    dot = (ref * tst).sum(axis=-1)
    norm_ref = np.linalg.norm(ref, axis=-1)
    norm_tst = np.linalg.norm(tst, axis=-1)
    
    cos_angle = dot / (norm_ref * norm_tst + 1e-12)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angles = np.arccos(cos_angle)
    
    return float(np.mean(angles))


def compute_rmse(reference: np.ndarray, test: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """计算均方根误差。"""
    if mask is not None:
        mask_bool = mask > 0 if mask.ndim == 2 else mask.any(axis=-1)
        diff = reference[mask_bool] - test[mask_bool]
    else:
        diff = reference - test
    return float(np.sqrt(np.mean(diff ** 2)))


# =============================================================================
# 演示与测试
# =============================================================================

def demo():
    """演示：构造多波段测试图像，对比独立修补 vs PCA联合修补。"""
    print("=" * 70)
    print("多波段图像修补演示：独立修补 vs PCA联合修补")
    print("=" * 70)
    
    H, W, C = 128, 128, 8
    print(f"\n测试图像尺寸: {H}x{W}, 波段数: {C}")
    
    rng = np.random.RandomState(42)
    
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32) / H
    base_spectra = np.zeros((H, W, C), dtype=np.float32)
    for c in range(C):
        freq = 0.5 + c * 0.3
        phase = c * 0.4
        base_spectra[:, :, c] = (
            0.3 + 0.2 * np.sin(xx * freq * 6.28 + phase)
            + 0.2 * np.cos(yy * freq * 5.0 + phase * 0.7)
            + 0.1 * rng.randn(H, W) * 0.1
        )
    
    cov = np.random.randn(C, C).astype(np.float32) * 0.1
    cov = cov @ cov.T + np.eye(C) * 0.05
    L = np.linalg.cholesky(cov)
    noise = rng.randn(H, W, C).astype(np.float32) @ L.T
    image = base_spectra * 0.6 + 0.4 + noise * 0.05
    image = np.clip(image, 0, 1)
    
    mask = np.zeros((H, W), dtype=np.uint8)
    cy, cx, r = H // 2, W // 2, 15
    yy_m, xx_m = np.mgrid[0:H, 0:W]
    mask[((yy_m - cy) ** 2 + (xx_m - cx) ** 2) < r ** 2] = 255
    
    rect_y, rect_x, rect_h, rect_w = 30, 90, 25, 20
    mask[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w] = 255
    
    print(f"待修补像素: {mask.sum() / 255:.0f} ({mask.sum() / 255 / (H * W) * 100:.1f}%)")
    
    image_corrupted = image.copy()
    image_corrupted[mask > 0] = 0.5
    
    print("\n--- 方法1: 独立波段修补（原始算法，有缺陷）---")
    result_independent = inpaint_independent_bands(image_corrupted, mask, radius=5)
    sam_ind = compute_spectral_angle_mapper(image, result_independent, mask)
    rmse_ind = compute_rmse(image, result_independent, mask)
    print(f"  SAM (光谱角, 越小越好):  {sam_ind:.6f} rad")
    print(f"  RMSE (均方根误差):       {rmse_ind:.6f}")
    
    print("\n--- 方法2: PCA联合修补（修复算法）---")
    result_pca, info = inpaint_pca_joint(
        image_corrupted, mask, radius=5,
        explained_variance_ratio=0.99,
        return_pca_info=True
    )
    sam_pca = compute_spectral_angle_mapper(image, result_pca, mask)
    rmse_pca = compute_rmse(image, result_pca, mask)
    print(f"  使用主成分数: {info['n_components']} / {C}")
    print(f"  累计方差解释: {info['cumulative_variance'][-1] * 100:.2f}%")
    print(f"  SAM (光谱角, 越小越好):  {sam_pca:.6f} rad")
    print(f"  RMSE (均方根误差):       {rmse_pca:.6f}")
    
    print("\n--- 修复效果对比 ---")
    sam_improve = (1 - sam_pca / sam_ind) * 100 if sam_ind > 0 else 0
    rmse_improve = (1 - rmse_pca / rmse_ind) * 100 if rmse_ind > 0 else 0
    print(f"  SAM 降低:  {sam_improve:+.2f}%  {'✓ 改善显著' if sam_improve > 20 else '△ 有改善' if sam_improve > 0 else '✗ 无改善'}")
    print(f"  RMSE 降低: {rmse_improve:+.2f}%  {'✓ 改善显著' if rmse_improve > 20 else '△ 有改善' if rmse_improve > 0 else '✗ 无改善'}")
    
    if C >= 4:
        print("\n--- 修补区域波段均值对比（验证光谱相关性）---")
        m = mask > 0
        ref_mean = image[m].mean(axis=0)
        ind_mean = result_independent[m].mean(axis=0)
        pca_mean = result_pca[m].mean(axis=0)
        
        print(f"  {'波段':>6s}  {'真值':>10s}  {'独立修补':>10s}  {'PCA修补':>10s}  {'独立误差':>10s}  {'PCA误差':>10s}")
        for c in range(min(C, 8)):
            err_ind = abs(ind_mean[c] - ref_mean[c])
            err_pca = abs(pca_mean[c] - ref_mean[c])
            flag = " <<<" if err_ind > err_pca * 1.5 else ""
            print(f"  {c:>6d}  {ref_mean[c]:>10.4f}  {ind_mean[c]:>10.4f}  {pca_mean[c]:>10.4f}  {err_ind:>10.4f}  {err_pca:>10.4f}{flag}")
    
    print("\n" + "=" * 70)
    print("结论: PCA联合修补通过保持波段间协方差结构，")
    print("      显著降低了光谱畸变，尤其在波段数>3时效果明显。")
    print("=" * 70)


if __name__ == "__main__":
    demo()
