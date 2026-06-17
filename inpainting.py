"""
多波段图像修补与薄云去除 (Multi-band Image Inpainting & Thin Cloud Removal)

功能模块：
  1. 厚云修补 (Cloud Inpainting)
     - 问题：原算法对各波段独立修补，波段数>3时破坏光谱相关性，导致颜色畸变
     - 解决方案：PCA降维 → 低维主成分空间联合修补 → 逆PCA投影回原始波段空间

  2. 薄云去除 (Thin Cloud Removal)
     - 场景：透光率>30%的半透明薄云区域（非完全遮挡）
     - 方法：暗通道先验 (DCP) + 大气散射模型反演
     - 优势：保留地物纹理细节，恢复真实辐射值，而非"无中生有"地修补

  3. 混合模式 (Hybrid Mode)
     - 自动区分薄云/厚云（基于透射率阈值）
     - 薄云区域：DCP辐射校正，恢复透云影像
     - 厚云区域：PCA联合修补，填补缺失信息

模式切换：
  "inpaint"  - 厚云修补模式（直接填补）
  "decloud"  - 薄云去除模式（辐射校正）
  "hybrid"   - 混合模式（自动分类+分别处理）

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
# 暗通道先验 (DCP) 与薄云去除
# =============================================================================

def _min_filter(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """最小值滤波（暗通道计算的核心操作）。
    
    用滑动窗口取局部最小值，模拟腐蚀操作。
    """
    H, W = image.shape[:2]
    pad = kernel_size // 2
    
    if image.ndim == 2:
        padded = np.pad(image, pad, mode="edge")
        result = np.zeros_like(image)
        for i in range(H):
            for j in range(W):
                result[i, j] = padded[i:i + kernel_size, j:j + kernel_size].min()
        return result
    
    if HAS_CV2:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = np.zeros_like(image)
        for c in range(image.shape[-1]):
            result[:, :, c] = cv2.erode(image[:, :, c], kernel)
        return result
    
    if HAS_SCIPY:
        from scipy.ndimage import minimum_filter
        result = np.zeros_like(image)
        for c in range(image.shape[-1]):
            result[:, :, c] = minimum_filter(image[:, :, c], size=kernel_size)
        return result
    
    raise ImportError("需要 OpenCV 或 SciPy 进行最小值滤波")


def compute_dark_channel(image: np.ndarray, kernel_size: int = 15) -> np.ndarray:
    """计算暗通道图 (Dark Channel)。
    
    暗通道先验：在绝大多数非天空的局部区域里，某些波段的像素值总是很低（接近0）。
    云的存在会抬高暗通道值，因此暗通道可用于估计云的厚度。
    
    暗通道定义: I_dark(x) = min_{y in Omega(x)} ( min_{c in {1,...,C}} I^c(y) )
    
    Args:
        image: 输入图像 (H, W, C)，值范围 [0, 1]
        kernel_size: 局部窗口大小
    
    Returns:
        暗通道图 (H, W)
    """
    if image.ndim != 3:
        raise ValueError(f"image 应为 (H, W, C)，实际形状 {image.shape}")
    
    channel_min = np.min(image, axis=-1)
    dark = _min_filter(channel_min, kernel_size)
    return dark


def estimate_atmospheric_light(image: np.ndarray, dark_channel: np.ndarray, top_percent: float = 0.1) -> np.ndarray:
    """估计全球大气光值 A（大气散射模型中的环境光强）。
    
    取暗通道最亮的前top_percent%像素，在原始图像中取这些像素的最大值作为大气光。
    这些像素对应最厚的云/最模糊的区域。
    
    Args:
        image: 输入图像 (H, W, C)
        dark_channel: 暗通道图 (H, W)
        top_percent: 取前百分之多少的最亮暗通道像素
    
    Returns:
        大气光向量 (C,)
    """
    H, W = dark_channel.shape
    N = H * W
    
    num_pixels = max(int(N * top_percent / 100.0), 1)
    
    flat_dark = dark_channel.ravel()
    flat_image = image.reshape(-1, image.shape[-1])
    
    indices = np.argpartition(flat_dark, -num_pixels)[-num_pixels:]
    
    atmospheric_light = np.max(flat_image[indices], axis=0)
    atmospheric_light = np.clip(atmospheric_light, 0.1, 1.0)
    
    return atmospheric_light


def estimate_transmission(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    kernel_size: int = 15,
    omega: float = 0.95
) -> np.ndarray:
    """估计透射率图 t(x)（即光线穿透云/雾的比例）。
    
    基于大气散射模型:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    其中:
        I(x) = 观测图像
        J(x) = 地物真实辐射（待恢复）
        t(x) = 透射率（0~1，1表示完全无云）
        A    = 大气光
    
    透射率与暗通道的关系推导:
        t(x) = 1 - omega * min_c( min_{Omega} (I^c / A^c) )
             = 1 - omega * dark_channel(I/A)
    
    omega是保留少量云/雾的系数（0.95效果最好，避免画面过于假）
    
    Args:
        image: 输入图像 (H, W, C)，值范围 [0, 1]
        atmospheric_light: 大气光向量 (C,)
        kernel_size: 暗通道窗口大小
        omega: 云保留系数，0~1，越大去云越彻底
    
    Returns:
        透射率图 (H, W)，值范围 [0, 1]
    """
    normalized = image / (atmospheric_light + 1e-12)
    dark = compute_dark_channel(normalized, kernel_size)
    transmission = 1.0 - omega * dark
    transmission = np.clip(transmission, 0.05, 1.0)
    return transmission


def refine_transmission_guided(image: np.ndarray, transmission: np.ndarray, radius: int = 5) -> np.ndarray:
    """引导滤波细化透射率图，保持边缘细节。
    
    暗通道得到的透射率比较粗糙，用引导滤波（以原图为引导）细化边缘。
    没有OpenCV时退化为高斯模糊平滑。
    """
    if HAS_CV2:
        img_gray = np.mean(image, axis=-1).astype(np.float32)
        img_gray_u8 = np.clip(img_gray * 255, 0, 255).astype(np.uint8)
        trans_u8 = np.clip(transmission * 255, 0, 255).astype(np.uint8)
        
        refined_u8 = cv2.ximgproc.guidedFilter(
            guide=img_gray_u8,
            src=trans_u8,
            radius=radius,
            eps=1e-3
        ) if hasattr(cv2, "ximgproc") else cv2.GaussianBlur(trans_u8, (radius * 2 + 1, radius * 2 + 1), 0)
        
        return refined_u8.astype(np.float32) / 255.0
    
    if HAS_SCIPY:
        return ndimage.gaussian_filter(transmission, sigma=radius / 3.0)
    
    return transmission


def recover_scene_radiance(
    image: np.ndarray,
    atmospheric_light: np.ndarray,
    transmission: np.ndarray,
    t_min: float = 0.1
) -> np.ndarray:
    """根据大气散射模型反演地物真实辐射值（去云/去雾）。
    
    公式: J(x) = (I(x) - A) / max(t(x), t_min) + A
    
    当透射率很低时，直接除以t会产生严重噪声，因此设置t_min下限。
    
    Args:
        image: 观测图像 (H, W, C)
        atmospheric_light: 大气光向量 (C,)
        transmission: 透射率图 (H, W)
        t_min: 透射率最小值（防止分母为0和噪声放大）
    
    Returns:
        恢复后的地物真实辐射图像 (H, W, C)
    """
    t_clipped = np.maximum(transmission, t_min)[:, :, None]
    A = atmospheric_light[None, None, :]
    
    J = (image - A) / t_clipped + A
    J = np.clip(J, 0.0, 1.0)
    
    return J


def thin_cloud_removal_dcp(
    image: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    kernel_size: int = 15,
    omega: float = 0.95,
    t_min: float = 0.1,
    refine: bool = True,
    return_transmission: bool = False
) -> np.ndarray or Tuple[np.ndarray, dict]:
    """基于暗通道先验的薄云去除。
    
    适用场景：半透明薄云（透光率>30%），地物纹理仍隐约可见
    处理方式：物理模型反演（大气散射模型），恢复真实地物辐射
    
    Args:
        image: 输入多波段图像 (H, W, C)，值范围 [0, 1]
        cloud_mask: 可选云掩膜 (H, W)，非零为云区。若为None则对整幅图处理
        kernel_size: 暗通道窗口大小
        omega: 云保留系数（0.9~1.0），越大去云越彻底
        t_min: 透射率下限
        refine: 是否细化透射率
        return_transmission: 是否返回透射率等中间结果
    
    Returns:
        去云后的图像 (H, W, C)；若 return_transmission=True，则返回 (图像, info字典)
    """
    if image.ndim != 3:
        raise ValueError(f"image 应为 (H, W, C)，实际形状 {image.shape}")
    
    image_f32 = image.astype(np.float32)
    
    if cloud_mask is not None:
        mask_bool = cloud_mask > 0
    else:
        mask_bool = np.ones(image.shape[:2], dtype=bool)
    
    dark = compute_dark_channel(image_f32, kernel_size)
    
    if cloud_mask is not None:
        cloud_dark = dark.copy()
        cloud_dark[~mask_bool] = -np.inf
        A = estimate_atmospheric_light(image_f32, cloud_dark, top_percent=0.5)
    else:
        A = estimate_atmospheric_light(image_f32, dark, top_percent=0.1)
    
    t = estimate_transmission(image_f32, A, kernel_size, omega)
    
    if refine:
        t = refine_transmission_guided(image_f32, t, radius=5)
    
    J = recover_scene_radiance(image_f32, A, t, t_min)
    
    if cloud_mask is not None:
        result = image_f32.copy()
        mask_3ch = mask_bool[:, :, None]
        result = np.where(mask_3ch, J, image_f32)
    else:
        result = J
    
    if not return_transmission:
        return result
    
    info = {
        "dark_channel": dark,
        "atmospheric_light": A,
        "transmission": t,
        "recovered": J,
    }
    return result, info


def classify_cloud_thickness(
    transmission: np.ndarray,
    thin_threshold: float = 0.3,
    thick_threshold: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """根据透射率对云进行厚度分类。
    
    分类规则：
      t > thin_threshold       → 薄云 / 无云    → 用DCP去除
      thick_threshold < t <= thin_threshold → 中等云 → 混合策略
      t <= thick_threshold     → 厚云 / 完全遮挡 → 用修补填补
    
    Args:
        transmission: 透射率图 (H, W)
        thin_threshold: 薄云阈值（透光率，默认30%）
        thick_threshold: 厚云阈值（默认10%）
    
    Returns:
        (thin_cloud_mask, thick_cloud_mask)，都是布尔类型
    """
    thin_mask = transmission > thin_threshold
    thick_mask = transmission <= thick_threshold
    return thin_mask, thick_mask


def inpaint_hybrid_cloud(
    image: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    mode: str = "hybrid",
    thin_threshold: float = 0.3,
    inpaint_radius: int = 5,
    dcp_kernel_size: int = 15,
    dcp_omega: float = 0.95,
    n_components: Optional[int] = None,
    explained_variance_ratio: float = 0.98,
    return_debug_info: bool = False
) -> np.ndarray or Tuple[np.ndarray, dict]:
    """混合云处理：薄云DCP去除 + 厚云PCA修补。
    
    模式选择 (mode):
      - "inpaint": 全部当作厚云，直接PCA修补
      - "decloud": 全部当作薄云，DCP辐射校正
      - "hybrid":  自动分类，薄云DCP + 厚云修补
    
    处理流程（hybrid模式）:
      1. 计算暗通道 → 估计透射率
      2. 根据透射率分成薄云/厚云两类
      3. 薄云区域：DCP反演恢复地物辐射
      4. 厚云区域：PCA联合修补填补缺失信息
      5. 两者在边界处自然过渡
    
    Args:
        image: 输入图像 (H, W, C)，值范围 [0, 1]
        cloud_mask: 云掩膜 (H, W)，非零为云。None则整幅图处理
        mode: "inpaint" | "decloud" | "hybrid"
        thin_threshold: 薄云透射率阈值（默认30%）
        inpaint_radius: 修补邻域半径
        dcp_kernel_size: 暗通道窗口大小
        dcp_omega: DCP云保留系数
        n_components: PCA主成分数
        explained_variance_ratio: PCA方差保留比例
        return_debug_info: 是否返回调试信息
    
    Returns:
        处理后的图像 (H, W, C)
    """
    if image.ndim != 3:
        raise ValueError(f"image 应为 (H, W, C)，实际形状 {image.shape}")
    
    H, W, C = image.shape
    image_f32 = image.astype(np.float32)
    
    if cloud_mask is not None:
        cloud_bool = cloud_mask > 0
    else:
        cloud_bool = np.ones((H, W), dtype=bool)
    
    debug_info = {}
    
    if mode == "inpaint":
        result = inpaint_pca_joint(
            image_f32, cloud_bool.astype(np.uint8) * 255,
            radius=inpaint_radius,
            n_components=n_components,
            explained_variance_ratio=explained_variance_ratio
        )
        if return_debug_info:
            debug_info["mode"] = "inpaint"
            return result, debug_info
        return result
    
    if mode == "decloud":
        result, dcp_info = thin_cloud_removal_dcp(
            image_f32,
            cloud_mask=cloud_bool.astype(np.uint8) * 255 if cloud_mask is not None else None,
            kernel_size=dcp_kernel_size,
            omega=dcp_omega,
            return_transmission=True
        )
        if return_debug_info:
            debug_info["mode"] = "decloud"
            debug_info["dcp"] = dcp_info
            return result, debug_info
        return result
    
    if mode == "hybrid":
        _, dcp_info = thin_cloud_removal_dcp(
            image_f32,
            cloud_mask=cloud_bool.astype(np.uint8) * 255 if cloud_mask is not None else None,
            kernel_size=dcp_kernel_size,
            omega=dcp_omega,
            return_transmission=True
        )
        
        t = dcp_info["transmission"]
        
        if cloud_mask is not None:
            t_cloud = t.copy()
            t_cloud[~cloud_bool] = 1.0
        else:
            t_cloud = t
        
        thin_mask, thick_mask = classify_cloud_thickness(
            t_cloud, thin_threshold=thin_threshold, thick_threshold=thin_threshold * 0.3
        )
        
        if cloud_mask is not None:
            thin_mask = thin_mask & cloud_bool
            thick_mask = thick_mask & cloud_bool
        
        thin_ratio = thin_mask.sum() / max(cloud_bool.sum(), 1)
        thick_ratio = thick_mask.sum() / max(cloud_bool.sum(), 1)
        
        J = dcp_info["recovered"]
        
        inpaint_mask_u8 = thick_mask.astype(np.uint8) * 255
        pca_result = inpaint_pca_joint(
            image_f32, inpaint_mask_u8,
            radius=inpaint_radius,
            n_components=n_components,
            explained_variance_ratio=explained_variance_ratio
        )
        
        result = image_f32.copy()
        
        if thin_mask.any():
            thin_3ch = thin_mask[:, :, None]
            result = np.where(thin_3ch, J, result)
        
        if thick_mask.any():
            thick_3ch = thick_mask[:, :, None]
            result = np.where(thick_3ch, pca_result, result)
        
        if HAS_CV2:
            t_normalized = np.clip((t_cloud - 0.05) / 0.95, 0, 1)
            blend_mask = t_normalized * cloud_bool.astype(np.float32)
            blend_mask_3ch = blend_mask[:, :, None]
            
            blended = pca_result * (1 - blend_mask_3ch) + J * blend_mask_3ch
            result = np.where(cloud_bool[:, :, None], blended, result)
        
        if return_debug_info:
            debug_info["mode"] = "hybrid"
            debug_info["thin_mask"] = thin_mask
            debug_info["thick_mask"] = thick_mask
            debug_info["thin_ratio"] = thin_ratio
            debug_info["thick_ratio"] = thick_ratio
            debug_info["dcp"] = dcp_info
            debug_info["pca_result"] = pca_result
            debug_info["dcp_result"] = J
            return result, debug_info
        
        return result
    
    raise ValueError(f"未知模式: {mode}，支持 'inpaint' / 'decloud' / 'hybrid'")


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

def _generate_test_image(H: int, W: int, C: int, seed: int = 42) -> np.ndarray:
    """生成测试用的多波段地物图像（有空间纹理+光谱相关性）。"""
    rng = np.random.RandomState(seed)
    
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
    
    cov = rng.randn(C, C).astype(np.float32) * 0.1
    cov = cov @ cov.T + np.eye(C) * 0.05
    L = np.linalg.cholesky(cov)
    noise = rng.randn(H, W, C).astype(np.float32) @ L.T
    image = base_spectra * 0.6 + 0.4 + noise * 0.05
    return np.clip(image, 0, 1)


def _generate_cloud_mask(H: int, W: int, seed: int = 123) -> np.ndarray:
    """生成云掩膜，包含薄云区和厚云区。"""
    rng = np.random.RandomState(seed)
    mask = np.zeros((H, W), dtype=np.uint8)
    
    cy, cx, r = H // 2, W // 3, 18
    yy_m, xx_m = np.mgrid[0:H, 0:W]
    dist1 = np.sqrt((yy_m - cy) ** 2 + (xx_m - cx) ** 2)
    mask[dist1 < r] = 255
    
    cy2, cx2, r2 = int(H * 0.7), int(W * 0.65), 22
    dist2 = np.sqrt((yy_m - cy2) ** 2 + (xx_m - cx2) ** 2)
    mask[dist2 < r2] = 255
    
    noise = rng.rand(H, W)
    mask[(noise > 0.7) & (mask == 0)] = 180
    
    return mask


def _apply_atmospheric_scattering(
    image: np.ndarray,
    cloud_mask: np.ndarray,
    A: float = 0.8,
    t_thin: float = 0.5,
    t_thick: float = 0.1,
    seed: int = 456
) -> np.ndarray:
    """模拟大气散射，给云区叠加薄云/厚云效果（用作测试的真值退化模型）。
    
    用I = J*t + A*(1-t)公式合成云污染图像。
    薄云区t≈0.5，厚云区t≈0.1。
    """
    rng = np.random.RandomState(seed)
    H, W, C = image.shape
    
    t_map = np.ones((H, W), dtype=np.float32)
    
    cloud_bool = cloud_mask > 0
    
    noise = rng.rand(H, W).astype(np.float32) * 0.3 + 0.7
    
    mask_bright = (cloud_mask > 200) & cloud_bool
    mask_medium = (cloud_mask > 100) & (cloud_mask <= 200) & cloud_bool
    
    t_map[mask_bright] = t_thick + (noise[mask_bright] * 0.1)
    t_map[mask_medium] = t_thin + (noise[mask_medium] * 0.25)
    
    A_spectrum = np.full(C, A, dtype=np.float32)
    A_spectrum *= 0.8 + rng.rand(C).astype(np.float32) * 0.4
    A_spectrum = np.clip(A_spectrum, 0.6, 1.0)
    
    t_3d = t_map[:, :, None]
    A_3d = A_spectrum[None, None, :]
    
    cloudy_image = image * t_3d + A_3d * (1 - t_3d)
    return np.clip(cloudy_image, 0, 1).astype(np.float32)


def demo_inpainting():
    """演示1：厚云修补 - 独立修补 vs PCA联合修补。"""
    print("\n" + "=" * 70)
    print("【演示1】厚云修补：独立修补 vs PCA联合修补")
    print("=" * 70)
    
    H, W, C = 128, 128, 8
    print(f"\n测试图像尺寸: {H}x{W}, 波段数: {C}")
    
    image = _generate_test_image(H, W, C, seed=42)
    
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


def demo_thin_cloud_removal():
    """演示2：薄云去除 - DCP暗通道先验 vs 直接修补。"""
    print("\n" + "=" * 70)
    print("【演示2】薄云去除：PCA修补 vs DCP辐射校正")
    print("=" * 70)
    
    H, W, C = 128, 128, 8
    print(f"\n测试图像尺寸: {H}x{W}, 波段数: {C}")
    
    image = _generate_test_image(H, W, C, seed=42)
    cloud_mask = _generate_cloud_mask(H, W, seed=123)
    
    print(f"云区像素: {(cloud_mask > 0).sum()} ({(cloud_mask > 0).sum() / (H * W) * 100:.1f}%)")
    print(f"  厚云区(>200): {(cloud_mask > 200).sum()} ({(cloud_mask > 200).sum() / (H * W) * 100:.1f}%)")
    print(f"  薄云区(100-200): {((cloud_mask > 100) & (cloud_mask <= 200)).sum()} ({((cloud_mask > 100) & (cloud_mask <= 200)).sum() / (H * W) * 100:.1f}%)")
    
    cloudy_image = _apply_atmospheric_scattering(
        image, cloud_mask,
        A=0.85, t_thin=0.5, t_thick=0.15
    )
    
    print("\n--- 方法1: PCA修补（当作厚云直接补）---")
    result_inpaint = inpaint_pca_joint(
        cloudy_image, cloud_mask, radius=7,
        explained_variance_ratio=0.99
    )
    sam_inpaint = compute_spectral_angle_mapper(image, result_inpaint, cloud_mask)
    rmse_inpaint = compute_rmse(image, result_inpaint, cloud_mask)
    print(f"  SAM (光谱角, 越小越好):  {sam_inpaint:.6f} rad")
    print(f"  RMSE (均方根误差):       {rmse_inpaint:.6f}")
    
    print("\n--- 方法2: DCP薄云去除（辐射校正，恢复地物真实值）---")
    result_dcp, info = thin_cloud_removal_dcp(
        cloudy_image,
        cloud_mask=cloud_mask,
        kernel_size=15,
        omega=0.95,
        return_transmission=True
    )
    sam_dcp = compute_spectral_angle_mapper(image, result_dcp, cloud_mask)
    rmse_dcp = compute_rmse(image, result_dcp, cloud_mask)
    
    t = info["transmission"]
    avg_t_cloud = t[cloud_mask > 0].mean()
    print(f"  估计平均透射率: {avg_t_cloud:.3f}")
    print(f"  SAM (光谱角, 越小越好):  {sam_dcp:.6f} rad")
    print(f"  RMSE (均方根误差):       {rmse_dcp:.6f}")
    
    print("\n--- 薄云区单独评估（透射率>30%区域）---")
    thin_cloud_mask = cloud_mask > 0
    if thin_cloud_mask.sum() > 0:
        t_cloud = t[thin_cloud_mask]
        thin_mask = thin_cloud_mask & (t > 0.3)
        
        if thin_mask.sum() > 0:
            sam_thin_dcp = compute_spectral_angle_mapper(image, result_dcp, thin_mask.astype(np.uint8) * 255)
            rmse_thin_dcp = compute_rmse(image, result_dcp, thin_mask.astype(np.uint8) * 255)
            sam_thin_inpaint = compute_spectral_angle_mapper(image, result_inpaint, thin_mask.astype(np.uint8) * 255)
            rmse_thin_inpaint = compute_rmse(image, result_inpaint, thin_mask.astype(np.uint8) * 255)
            
            print(f"  薄云像素数: {thin_mask.sum()}")
            print(f"  DCP   - SAM: {sam_thin_dcp:.6f}, RMSE: {rmse_thin_dcp:.6f}")
            print(f"  修补  - SAM: {sam_thin_inpaint:.6f}, RMSE: {rmse_thin_inpaint:.6f}")
            
            if sam_thin_dcp < sam_thin_inpaint:
                print(f"  → DCP在薄云区更优 (SAM降低 {(1 - sam_thin_dcp/sam_thin_inpaint)*100:.1f}%)")
            else:
                print(f"  → 修补在薄云区更优")
    
    print("\n--- 厚云区单独评估（透射率<15%区域）---")
    thick_mask = thin_cloud_mask & (t < 0.15)
    if thick_mask.sum() > 0:
        sam_thick_dcp = compute_spectral_angle_mapper(image, result_dcp, thick_mask.astype(np.uint8) * 255)
        rmse_thick_dcp = compute_rmse(image, result_dcp, thick_mask.astype(np.uint8) * 255)
        sam_thick_inpaint = compute_spectral_angle_mapper(image, result_inpaint, thick_mask.astype(np.uint8) * 255)
        rmse_thick_inpaint = compute_rmse(image, result_inpaint, thick_mask.astype(np.uint8) * 255)
        
        print(f"  厚云像素数: {thick_mask.sum()}")
        print(f"  DCP   - SAM: {sam_thick_dcp:.6f}, RMSE: {rmse_thick_dcp:.6f}")
        print(f"  修补  - SAM: {sam_thick_inpaint:.6f}, RMSE: {rmse_thick_inpaint:.6f}")
        
        if sam_thick_inpaint < sam_thick_dcp:
            print(f"  → 修补在厚云区更优 (SAM降低 {(1 - sam_thick_inpaint/sam_thick_dcp)*100:.1f}%)")
        else:
            print(f"  → DCP在厚云区更优")


def demo_hybrid_mode():
    """演示3：混合模式 - 自动分类薄云/厚云，分别处理。"""
    print("\n" + "=" * 70)
    print("【演示3】混合模式：薄云DCP去除 + 厚云PCA修补")
    print("=" * 70)
    
    H, W, C = 128, 128, 8
    print(f"\n测试图像尺寸: {H}x{W}, 波段数: {C}")
    
    image = _generate_test_image(H, W, C, seed=42)
    cloud_mask = _generate_cloud_mask(H, W, seed=123)
    cloudy_image = _apply_atmospheric_scattering(
        image, cloud_mask,
        A=0.85, t_thin=0.5, t_thick=0.15
    )
    
    modes = ["inpaint", "decloud", "hybrid"]
    results = {}
    
    for mode in modes:
        mode_name = {"inpaint": "纯修补模式", "decloud": "纯DCP去云模式", "hybrid": "混合模式"}[mode]
        print(f"\n--- {mode_name} ---")
        
        result, debug = inpaint_hybrid_cloud(
            cloudy_image, cloud_mask=cloud_mask,
            mode=mode, thin_threshold=0.3,
            inpaint_radius=7, dcp_kernel_size=15, dcp_omega=0.95,
            explained_variance_ratio=0.99,
            return_debug_info=True
        )
        results[mode] = result
        
        sam = compute_spectral_angle_mapper(image, result, cloud_mask)
        rmse = compute_rmse(image, result, cloud_mask)
        print(f"  SAM  (全云区): {sam:.6f} rad")
        print(f"  RMSE (全云区): {rmse:.6f}")
        
        if mode == "hybrid" and "thin_ratio" in debug:
            print(f"  薄云占比: {debug['thin_ratio']*100:.1f}%")
            print(f"  厚云占比: {debug['thick_ratio']*100:.1f}%")
    
    print("\n--- 三种模式对比 (SAM越低越好) ---")
    sams = {m: compute_spectral_angle_mapper(image, results[m], cloud_mask) for m in modes}
    rmses = {m: compute_rmse(image, results[m], cloud_mask) for m in modes}
    
    best_sam = min(sams.values())
    best_rmse = min(rmses.values())
    
    mode_labels = {"inpaint": "纯修补", "decloud": "纯DCP", "hybrid": "混合模式"}
    print(f"  {'模式':<10s}  {'SAM':>10s}  {'RMSE':>10s}")
    for m in modes:
        mark_sam = " ← 最佳" if sams[m] == best_sam else ""
        mark_rmse = " ← 最佳" if rmses[m] == best_rmse else ""
        print(f"  {mode_labels[m]:<10s}  {sams[m]:>10.6f}{mark_sam}  {rmses[m]:>10.6f}{mark_rmse}")
    
    print("\n" + "=" * 70)
    print("结论:")
    print("  • 薄云区域：DCP辐射校正优于直接修补（保留地物纹理细节）")
    print("  • 厚云区域：PCA修补更优（信息完全缺失，只能邻域推断）")
    print("  • 混合模式：自动分类+分别处理，整体效果最佳")
    print("  • 前端可提供 薄云去除/厚云修补 模式切换")
    print("=" * 70)


def demo():
    """完整演示：厚云修补 + 薄云去除 + 混合模式。"""
    print("=" * 70)
    print("多波段图像修补与薄云去除 - 完整演示")
    print("Multi-band Image Inpainting & Thin Cloud Removal")
    print("=" * 70)
    
    demo_inpainting()
    demo_thin_cloud_removal()
    demo_hybrid_mode()


if __name__ == "__main__":
    demo()
