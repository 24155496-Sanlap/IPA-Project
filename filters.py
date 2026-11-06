
import cv2
import numpy as np


def validate_image(image):
    if not isinstance(image, np.ndarray):
        raise TypeError("Expected a NumPy image array.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    if image.dtype != np.uint8:
        raise TypeError("Expected uint8 pixels.")
    return image


def blend_images(original, filtered, strength=1.0):
    validate_image(original)
    validate_image(filtered)

    if original.shape != filtered.shape:
        raise ValueError("Image dimensions must match.")

    strength = float(np.clip(strength, 0.0, 1.0))
    result = (
        original.astype(np.float32) * (1 - strength)
        + filtered.astype(np.float32) * strength
    )
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignette(
    image,
    strength=0.5,
    radius=0.65,
    softness=0.45,
    center=(0.5, 0.5)
):
    validate_image(image)

    if softness <= 0:
        raise ValueError("softness must be greater than zero.")
    if radius < 0:
        raise ValueError("radius must be non-negative.")

    h, w = image.shape[:2]

    # Express coordinates in units relative to image size.
    scale = min(h, w) / 2.0
    cx, cy = center[0] * (w - 1), center[1] * (h - 1)

    x = (np.arange(w, dtype=np.float32) - cx) / scale
    y = (np.arange(h, dtype=np.float32) - cy) / scale
    xx, yy = np.meshgrid(x, y)

    distance = np.sqrt(xx**2 + yy**2)

    # Smooth transition from the bright center to dark edges.
    t = np.clip((distance - radius) / softness, 0, 1)
    smooth = t * t * (3 - 2 * t)

    mask = 1.0 - np.clip(strength, 0, 1) * smooth
    result = image.astype(np.float32) * mask[..., None]

    return np.clip(result, 0, 255).astype(np.uint8)



def apply_sepia(image, strength=1.0):
    validate_image(image)

    # Matrix assumes RGB channel order.
    matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)

    sepia = image.astype(np.float32) @ matrix.T
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)

    return blend_images(image, sepia, strength)



def apply_monochrome(
    image,
    strength=1.0,
    contrast=1.0,
    brightness=0,
    tint=(255, 255, 255)
):
    validate_image(image)

    if contrast < 0:
        raise ValueError("contrast must be non-negative.")

    rgb = image.astype(np.float32)

    # Perceptual grayscale approximation.
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    # Adjust contrast around mid-gray and brightness in pixel units.
    gray = (gray - 127.5) * contrast + 127.5 + brightness
    gray = np.clip(gray, 0, 255)

    # Optional tint: white gives neutral gray.
    tint_rgb = np.asarray(tint, dtype=np.float32)
    if tint_rgb.shape != (3,):
        raise ValueError("tint must contain three RGB values.")

    tinted = gray[:, :, None] * (tint_rgb / 255.0)
    tinted = np.clip(tinted, 0, 255).astype(np.uint8)

    return blend_images(image, tinted, strength)



def apply_warm_cool(image, temperature=0, strength=1.0):
    validate_image(image)

    # Temperature ranges from -100 (cool) to +100 (warm).
    t = float(np.clip(temperature, -100, 100)) / 100.0

    rgb = image.astype(np.float32).copy()

    # RGB order: channel 0 = red, channel 2 = blue.
    rgb[:, :, 0] *= 1.0 + 0.35 * t
    rgb[:, :, 1] *= 1.0 + 0.04 * abs(t)
    rgb[:, :, 2] *= 1.0 - 0.35 * t

    adjusted = np.clip(rgb, 0, 255).astype(np.uint8)
    return blend_images(image, adjusted, strength)


def apply_vintage(
    image,
    strength=1.0,
    warmth=25,
    fade=0.15,
    contrast=0.90,
    saturation=0.75,
    grain=5,
    vignette=0.30,
    seed=42
):
    validate_image(image)

    if contrast < 0 or saturation < 0:
        raise ValueError("contrast and saturation must be non-negative.")
    if grain < 0:
        raise ValueError("grain must be non-negative.")

    # Work on a float copy; never modify the input.
    result = image.astype(np.float32).copy()

    # 1. Warm the image.
    result = apply_warm_cool(
        np.clip(result, 0, 255).astype(np.uint8), temperature=warmth
    ).astype(np.float32)

    # 2. Fade black levels.
    fade = float(np.clip(fade, 0, 1))
    result = result * (1 - fade) + 35 * fade

    # 3. Change contrast around middle gray.
    result = (result - 127.5) * contrast + 127.5

    # 4. Adjust saturation around grayscale luminance.
    gray = (
        0.299 * result[:, :, 0]
        + 0.587 * result[:, :, 1]
        + 0.114 * result[:, :, 2]
    )
    result = (
        gray[:, :, None]
        + saturation * (result - gray[:, :, None])
    )

    result = np.clip(result, 0, 255).astype(np.uint8)

    # 5. Add reproducible film grain.
    if grain > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, grain, result.shape[:2])
        result = np.clip(
            result.astype(np.float32) + noise[:, :, None],
            0, 255
        ).astype(np.uint8)

    # 6. Add vignette.
    if vignette > 0:
        result = apply_vignette(
            result, strength=vignette,
            radius=0.60, softness=0.55
        )

    return blend_images(image, result, strength)



def apply_negative(image, strength=1.0):
    validate_image(image)

    negative = 255 - image
    return blend_images(image, negative, strength)


def apply_posterization(image, levels=6, strength=1.0):
    validate_image(image)

    levels = int(levels)
    if not 2 <= levels <= 256:
        raise ValueError("levels must be between 2 and 256.")

    step = 255.0 / (levels - 1)

    posterized = np.round(
        image.astype(np.float32) / step
    ) * step

    posterized = np.clip(posterized, 0, 255).astype(np.uint8)

    return blend_images(image, posterized, strength)


def apply_dreamy_glow(
    image,
    strength=0.7,
    threshold=170,
    blur_radius=15
):
    validate_image(image)

    if blur_radius <= 0:
        raise ValueError("blur_radius must be positive.")

    rgb = image.astype(np.float32)

    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    # Keep primarily the bright regions.
    mask = np.clip(
        (gray - threshold) / max(255 - threshold, 1),
        0, 1
    )

    highlights = rgb * mask[:, :, None]

    glow = cv2.GaussianBlur(
        highlights,
        (0, 0),
        sigmaX=blur_radius,
        sigmaY=blur_radius
    )

    result = np.clip(
        rgb + np.clip(strength, 0, 2) * glow,
        0, 255
    ).astype(np.uint8)

    return result


def apply_emboss(
    image,
    strength=1.0,
    depth=1.0,
    direction="diagonal"
):
    validate_image(image)

    kernels = {
        "diagonal": np.array([
            [-2, -1, 0],
            [-1,  1, 1],
            [ 0,  1, 2]
        ], dtype=np.float32),

        "horizontal": np.array([
            [-1, 0, 1],
            [-1, 1, 1],
            [-1, 0, 1]
        ], dtype=np.float32),

        "vertical": np.array([
            [-1, -1, -1],
            [ 0,  1,  0],
            [ 1,  1,  1]
        ], dtype=np.float32)
    }

    if direction not in kernels:
        raise ValueError(
            "direction must be diagonal, horizontal, or vertical."
        )

    # Apply the kernel separately to each RGB channel.
    embossed = cv2.filter2D(
        image.astype(np.float32),
        -1,
        kernels[direction] * depth
    )

    # Center the relief around middle gray.
    embossed = np.clip(128 + embossed, 0, 255).astype(np.uint8)

    return blend_images(image, embossed, strength)



def apply_pencil_sketch(
    image,
    strength=1.0,
    blur_radius=15,
    darkness=1.0,
    edge_strength=0.25
):
    validate_image(image)

    if blur_radius <= 0:
        raise ValueError("blur_radius must be positive.")

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    inverted = 255 - gray
    blurred = cv2.GaussianBlur(
        inverted, (0, 0), sigmaX=blur_radius
    )

    # Color-dodge style sketch.
    denominator = 255.0 - blurred.astype(np.float32)
    sketch = gray.astype(np.float32) * 255.0 / np.maximum(
        denominator, 1.0
    )

    sketch = np.clip(sketch, 0, 255)

    # darkness below 1 makes the sketch lighter.
    sketch = 255 - (255 - sketch) * np.clip(darkness, 0, 2)
    sketch = np.clip(sketch, 0, 255)

    # Optionally strengthen structural edges.
    edges = cv2.Canny(gray, 70, 150)
    edge_mask = edges.astype(np.float32) / 255.0

    sketch *= (
        1.0 - np.clip(edge_strength, 0, 1)
        * edge_mask
    )

    sketch_rgb = np.repeat(
        sketch[:, :, None], 3, axis=2
    ).astype(np.uint8)

    return blend_images(image, sketch_rgb, strength)



def apply_cartoon(
    image,
    strength=1.0,
    color_levels=8,
    smoothing=9,
    outline_strength=0.8,
    outline_threshold=100
):
    validate_image(image)

    if not 2 <= int(color_levels) <= 256:
        raise ValueError("color_levels must be between 2 and 256.")

    if smoothing < 1:
        raise ValueError("smoothing must be positive.")

    # Bilateral filtering reduces texture while preserving boundaries.
    smooth = cv2.bilateralFilter(
        image,
        d=9,
        sigmaColor=float(smoothing) * 10,
        sigmaSpace=float(smoothing) * 2
    )

    # Reduce the number of levels in each RGB channel.
    step = 255.0 / (int(color_levels) - 1)
    flat_colors = np.round(
        smooth.astype(np.float32) / step
    ) * step
    flat_colors = np.clip(flat_colors, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Find edges and thicken them slightly.
    edges = cv2.Canny(
        gray,
        max(0, int(outline_threshold * 0.5)),
        max(1, int(outline_threshold))
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    edge_mask = edges.astype(np.float32) / 255.0
    edge_mask *= np.clip(outline_strength, 0, 1)

    # Blend detected edges toward black.
    result = flat_colors.astype(np.float32)
    result *= (1 - edge_mask[:, :, None])

    result = np.clip(result, 0, 255).astype(np.uint8)

    return blend_images(image, result, strength)



def apply_cinematic(
    image,
    strength=1.0,
    shadow_tint=(20, 105, 125),
    highlight_tint=(255, 165, 85),
    shadow_amount=0.20,
    highlight_amount=0.16,
    contrast=1.12,
    saturation=1.08
):
    validate_image(image)

    if contrast < 0 or saturation < 0:
        raise ValueError("contrast and saturation must be non-negative.")

    rgb = image.astype(np.float32)

    luminance = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )

    def smoothstep(low, high, values):
        t = np.clip((values - low) / (high - low), 0, 1)
        return t * t * (3 - 2 * t)

    # Shadows are strongest at low luminance.
    shadows = 1.0 - smoothstep(35, 185, luminance)

    # Highlights are strongest at high luminance.
    highlights = smoothstep(75, 235, luminance)

    shadow_color = np.asarray(shadow_tint, dtype=np.float32)
    highlight_color = np.asarray(highlight_tint, dtype=np.float32)

    # Blend the shadow tint into the shadow regions.
    sw = shadows[:, :, None] * np.clip(shadow_amount, 0, 1)
    rgb = rgb * (1 - sw) + shadow_color * sw

    # Blend the highlight tint into the bright regions.
    hw = highlights[:, :, None] * np.clip(highlight_amount, 0, 1)
    rgb = rgb * (1 - hw) + highlight_color * hw

    # Adjust contrast.
    rgb = (rgb - 127.5) * contrast + 127.5

    # Adjust saturation around luminance.
    gray = (
        0.299 * rgb[:, :, 0]
        + 0.587 * rgb[:, :, 1]
        + 0.114 * rgb[:, :, 2]
    )
    rgb = gray[:, :, None] + saturation * (
        rgb - gray[:, :, None]
    )

    graded = np.clip(rgb, 0, 255).astype(np.uint8)

    return blend_images(image, graded, strength)




# ---------------------------------------------------------------------------
# Edge detection and outline effects
# ---------------------------------------------------------------------------

def _odd_kernel(value, minimum=1, maximum=31):
    """Return a valid odd OpenCV kernel size."""
    value = int(np.clip(int(value), minimum, maximum))
    return value if value % 2 else min(value + 1, maximum)


def apply_canny_edge_detection(
    image, low_threshold=50, high_threshold=150, blur_size=3, aperture_size=3
):
    validate_image(image)
    low, high = sorted((int(np.clip(low_threshold, 0, 255)),
                        int(np.clip(high_threshold, 0, 255))))
    if low == high:
        high = min(255, low + 1)
    blur_size = _odd_kernel(blur_size, 1, 31)
    aperture_size = int(aperture_size)
    if aperture_size not in (3, 5, 7):
        raise ValueError("aperture_size must be 3, 5, or 7.")
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if blur_size > 1:
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    edges = cv2.Canny(gray, low, high, apertureSize=aperture_size, L2gradient=True)
    return np.repeat(edges[:, :, None], 3, axis=2)


def apply_sobel_filter(image, dx=1, dy=1, kernel_size=3, strength=1.0):
    validate_image(image)
    dx, dy = int(dx), int(dy)
    if dx < 0 or dy < 0 or (dx == 0 and dy == 0) or dx + dy > 2:
        raise ValueError("Choose dx/dy as (1,0), (0,1), or (1,1).")
    kernel_size = _odd_kernel(kernel_size, 1, 7)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, dx, 0, ksize=kernel_size) if dx else np.zeros_like(gray, dtype=np.float32)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, dy, ksize=kernel_size) if dy else np.zeros_like(gray, dtype=np.float32)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak > 0:
        magnitude = magnitude * (255.0 / peak)
    edge = np.clip(magnitude * float(np.clip(strength, 0, 5)), 0, 255).astype(np.uint8)
    return np.repeat(edge[:, :, None], 3, axis=2)


def apply_laplacian_edge_detection(image, kernel_size=3, scale=1.0, noise_reduction=3):
    validate_image(image)
    kernel_size = _odd_kernel(kernel_size, 1, 7)
    noise_reduction = _odd_kernel(noise_reduction, 1, 31)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if noise_reduction > 1:
        gray = cv2.GaussianBlur(gray, (noise_reduction, noise_reduction), 0)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel_size, scale=float(scale))
    lap = np.abs(lap)
    peak = float(lap.max())
    if peak > 0:
        lap *= 255.0 / peak
    edge = np.clip(lap, 0, 255).astype(np.uint8)
    return np.repeat(edge[:, :, None], 3, axis=2)


def apply_edge_overlay(
    image, detector="canny", outline_color_r=255, outline_color_g=40,
    outline_color_b=20, thickness=1, opacity=0.85, threshold=100
):
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    detector = str(detector).lower()
    threshold = int(np.clip(threshold, 0, 255))
    if detector == "canny":
        edges = cv2.Canny(gray, max(0, threshold // 2), max(1, threshold))
    elif detector == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        edges = (mag >= threshold).astype(np.uint8) * 255
    elif detector == "laplacian":
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        edges = (lap >= threshold).astype(np.uint8) * 255
    else:
        raise ValueError("detector must be canny, sobel, or laplacian.")
    thickness = int(np.clip(thickness, 1, 15))
    if thickness > 1:
        edges = cv2.dilate(edges, np.ones((thickness, thickness), np.uint8), iterations=1)
    color = np.array([outline_color_r, outline_color_g, outline_color_b], dtype=np.float32)
    color = np.clip(color, 0, 255)
    alpha = (edges.astype(np.float32) / 255.0 * float(np.clip(opacity, 0, 1)))[:, :, None]
    result = image.astype(np.float32) * (1 - alpha) + color[None, None, :] * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_outline_contour_art(image, threshold=100, line_thickness=2, inversion=False, smoothness=3):
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    smoothness = _odd_kernel(smoothness, 1, 31)
    if smoothness > 1:
        gray = cv2.GaussianBlur(gray, (smoothness, smoothness), 0)
    threshold = int(np.clip(threshold, 0, 255))
    _, edges = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # Use morphological gradient to outline shapes rather than fill regions.
    edges = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    _, edges = cv2.threshold(edges, threshold, 255, cv2.THRESH_BINARY)
    line_thickness = int(np.clip(line_thickness, 1, 15))
    if line_thickness > 1:
        edges = cv2.dilate(edges, np.ones((line_thickness, line_thickness), np.uint8), iterations=1)
    if inversion:
        edges = 255 - edges
    return np.repeat(edges[:, :, None], 3, axis=2)


# ---------------------------------------------------------------------------
# Blur effects
# ---------------------------------------------------------------------------

def apply_gaussian_blur(image, kernel_size=9, sigma=2.0, strength=1.0):
    validate_image(image)
    kernel_size = _odd_kernel(kernel_size, 1, 101)
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=max(0.0, float(sigma)))
    return blend_images(image, blurred, strength)


def apply_motion_blur(image, length=15, angle=0.0, strength=1.0):
    validate_image(image)
    length = int(np.clip(length, 1, 101))
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    radians = np.deg2rad(float(angle))
    dx, dy = np.cos(radians) * center, np.sin(radians) * center
    p1 = (int(round(center - dx)), int(round(center - dy)))
    p2 = (int(round(center + dx)), int(round(center + dy)))
    cv2.line(kernel, p1, p2, 1.0, thickness=1)
    total = float(kernel.sum())
    if total <= 0:
        kernel[length // 2, length // 2] = 1.0
    else:
        kernel /= total
    blurred = cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return blend_images(image, blurred, strength)


def apply_background_bokeh_blur(image, blur_radius=21, mask=None, feather=35, background_strength=1.0):
    """Blur the background using a supplied subject mask or a soft centered ellipse.

    A mask may be a 2-D NumPy array (nonzero means subject) with the same H/W
    dimensions as the image. Without one, a centered elliptical subject estimate
    is used; this is a fallback, not AI segmentation.
    """
    validate_image(image)
    h, w = image.shape[:2]
    radius = _odd_kernel(blur_radius, 1, 101)
    blurred = cv2.GaussianBlur(image, (radius, radius), 0)
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.ndim != 2 or mask.shape != (h, w):
            raise ValueError("mask must be a 2-D NumPy array matching the image dimensions.")
        subject = (mask.astype(np.float32) > 0).astype(np.float32)
        feather_px = max(0, int(feather))
        if feather_px:
            feather_kernel = _odd_kernel(min(feather_px * 2 + 1, 101), 1, 101)
            subject = cv2.GaussianBlur(subject, (feather_kernel, feather_kernel), 0)
            subject = np.clip(subject, 0, 1)
    else:
        # Soft elliptical center mask is a predictable fallback, not semantic segmentation.
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx - (w - 1) / 2) / max(w * 0.42, 1)
        ny = (yy - (h - 1) / 2) / max(h * 0.42, 1)
        distance = np.sqrt(nx * nx + ny * ny)
        softness = max(float(feather) / max(min(h, w), 1), 0.001)
        subject = np.clip((1.15 - distance) / softness + 0.5, 0, 1)
    background = (1 - subject) * float(np.clip(background_strength, 0, 1))
    result = image.astype(np.float32) * (1 - background[:, :, None]) + blurred.astype(np.float32) * background[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_radial_zoom_blur(image, center_x=0.5, center_y=0.5, amount=0.08, iterations=8, strength=1.0):
    validate_image(image)
    h, w = image.shape[:2]
    cx = float(np.clip(center_x, 0, 1)) * (w - 1)
    cy = float(np.clip(center_y, 0, 1)) * (h - 1)
    amount = float(np.clip(amount, 0, 0.5))
    iterations = int(np.clip(iterations, 1, 30))
    yy, xx = np.indices((h, w), dtype=np.float32)
    accum = np.zeros_like(image, dtype=np.float32)
    for i in range(iterations):
        factor = 1.0 - amount * (i + 1) / iterations
        map_x = cx + (xx - cx) * factor
        map_y = cy + (yy - cy) * factor
        accum += cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT).astype(np.float32)
    zoomed = np.clip(accum / iterations, 0, 255).astype(np.uint8)
    return blend_images(image, zoomed, strength)


def apply_tilt_shift_blur(image, focus_position=0.5, focus_width=0.25, transition=0.2, blur_strength=1.0):
    validate_image(image)
    h, w = image.shape[:2]
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=max(1.0, min(h, w) * 0.025))
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    distance = np.abs(y - float(np.clip(focus_position, 0, 1)))
    half_width = max(float(focus_width), 0.01) / 2
    transition = max(float(transition), 0.001)
    blur_mask = np.clip((distance - half_width) / transition, 0, 1)
    blur_mask = blur_mask * blur_mask * (3 - 2 * blur_mask)
    blur_mask = blur_mask * float(np.clip(blur_strength, 0, 1))
    result = image.astype(np.float32) * (1 - blur_mask[:, :, None]) + blurred.astype(np.float32) * blur_mask[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_median_blur(image, kernel_size=5, strength=1.0):
    validate_image(image)
    kernel_size = _odd_kernel(kernel_size, 3, 31)
    blurred = cv2.medianBlur(image, kernel_size)
    return blend_images(image, blurred, strength)


# ---------------------------------------------------------------------------
# Additional enhancement and analysis effects
# ---------------------------------------------------------------------------

def apply_sharpen_unsharp_mask(image, amount=1.5, radius=2.0, threshold=5):
    validate_image(image)
    radius = float(np.clip(radius, 0.1, 25))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=radius)
    original = image.astype(np.float32)
    detail = original - blurred.astype(np.float32)
    mask = np.max(np.abs(detail), axis=2) >= int(np.clip(threshold, 0, 255))
    result = original + float(np.clip(amount, 0, 5)) * detail * mask[:, :, None]
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_clahe_contrast(image, clip_limit=2.0, tile_grid_size=8, strength=1.0):
    validate_image(image)
    clip_limit = float(np.clip(clip_limit, 0.1, 20))
    grid = int(np.clip(tile_grid_size, 2, 32))
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid, grid))
    enhanced = cv2.merge((clahe.apply(l), a, b))
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
    return blend_images(image, enhanced, strength)


def apply_histogram_equalization(image, mode="luminance", strength=1.0, channel="all"):
    validate_image(image)
    mode = str(mode).lower()
    channel = str(channel).lower()
    if mode == "luminance":
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        lab_eq = cv2.merge((cv2.equalizeHist(l), a, b))
        enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    elif mode == "channel":
        enhanced = image.copy()
        channels = list(range(3)) if channel == "all" else [{"red": 0, "green": 1, "blue": 2}.get(channel, -1)]
        if -1 in channels:
            raise ValueError("channel must be all, red, green, or blue.")
        for index in channels:
            enhanced[:, :, index] = cv2.equalizeHist(enhanced[:, :, index])
    else:
        raise ValueError("mode must be luminance or channel.")
    return blend_images(image, enhanced, strength)


def apply_thresholding(image, threshold=128, method="binary", block_size=11, constant=2):
    validate_image(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    method = str(method).lower()
    threshold = int(np.clip(threshold, 0, 255))
    if method == "binary":
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    elif method == "binary_inverse":
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    elif method == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        block_size = _odd_kernel(block_size, 3, 99)
        kind = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        out = cv2.adaptiveThreshold(gray, 255, kind, cv2.THRESH_BINARY, block_size, float(constant))
    else:
        raise ValueError("Unsupported thresholding method.")
    return np.repeat(out[:, :, None], 3, axis=2)


def apply_fourier_frequency_filter(image, filter_type="lowpass", cutoff=30, order=2, notch_frequency=40):
    validate_image(image)
    filter_type = str(filter_type).lower()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = gray.shape
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    yy, xx = np.ogrid[:h, :w]
    distance = np.sqrt((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
    cutoff = max(1.0, float(cutoff))
    if filter_type == "lowpass":
        mask = 1 / (1 + (distance / cutoff) ** (2 * int(np.clip(order, 1, 8))))
    elif filter_type == "highpass":
        mask = 1 - 1 / (1 + (distance / cutoff) ** (2 * int(np.clip(order, 1, 8))))
    elif filter_type == "notch":
        # Suppress a narrow ring around the requested radial frequency.
        notch = max(1.0, float(notch_frequency))
        mask = np.clip(np.abs(distance - notch) / max(notch * 0.08, 1), 0, 1)
    else:
        raise ValueError("filter_type must be lowpass, highpass, or notch.")
    filtered = np.fft.ifft2(np.fft.ifftshift(spectrum * mask)).real
    lo, hi = float(filtered.min()), float(filtered.max())
    out = np.zeros_like(filtered, dtype=np.uint8) if hi <= lo else np.clip((filtered - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)
    return np.repeat(out[:, :, None], 3, axis=2)


def apply_edge_preserving_denoising(image, strength=5, spatial_radius=7, color_sigma=50):
    validate_image(image)
    diameter = int(np.clip(round(float(spatial_radius)), 1, 25))
    sigma_color = float(np.clip(color_sigma, 1, 150))
    sigma_space = float(np.clip(strength, 1, 150))
    result = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
    return result


name_function_map = {
    # Creative filters
    "vignette": apply_vignette,
    "sepia": apply_sepia,
    "monochrome": apply_monochrome,
    "warm/cool": apply_warm_cool,
    "vintage/retro": apply_vintage,
    "negative": apply_negative,
    "posterization": apply_posterization,
    "dreamy glow": apply_dreamy_glow,
    "emboss": apply_emboss,
    "pencil sketch": apply_pencil_sketch,
    "cartoon": apply_cartoon,
    "cinematic": apply_cinematic,
    # Edge detection and outline effects
    "canny edge detection": apply_canny_edge_detection,
    "sobel filter": apply_sobel_filter,
    "laplacian edge detection": apply_laplacian_edge_detection,
    "edge overlay": apply_edge_overlay,
    "outline contour art": apply_outline_contour_art,
    # Blur effects
    "gaussian blur": apply_gaussian_blur,
    "motion blur": apply_motion_blur,
    "background bokeh blur": apply_background_bokeh_blur,
    "radial zoom blur": apply_radial_zoom_blur,
    "tilt-shift blur": apply_tilt_shift_blur,
    "median blur": apply_median_blur,
    # Enhancement and analysis
    "sharpening / unsharp mask": apply_sharpen_unsharp_mask,
    "clahe contrast": apply_clahe_contrast,
    "histogram equalization": apply_histogram_equalization,
    "thresholding": apply_thresholding,
    "fourier frequency filter": apply_fourier_frequency_filter,
    "edge-preserving denoising": apply_edge_preserving_denoising,
}

FILTER_CATEGORIES = {
    "Creative filters": [
        "vignette", "sepia", "monochrome", "warm/cool", "vintage/retro",
        "negative", "posterization", "dreamy glow", "emboss", "pencil sketch",
        "cartoon", "cinematic",
    ],
    "Edge detection and outline effects": [
        "canny edge detection", "sobel filter", "laplacian edge detection",
        "edge overlay", "outline contour art",
    ],
    "Blur effects": [
        "gaussian blur", "motion blur", "background bokeh blur",
        "radial zoom blur", "tilt-shift blur", "median blur",
    ],
    "Enhancement and analysis": [
        "sharpening / unsharp mask", "clahe contrast", "histogram equalization",
        "thresholding", "fourier frequency filter", "edge-preserving denoising",
    ],
}
