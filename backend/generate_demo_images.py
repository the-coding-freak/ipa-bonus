import os
import cv2
import numpy as np
import warnings

# Suppress warnings from skimage
warnings.filterwarnings('ignore')

try:
    import tifffile
except ImportError:
    print("Please install tifffile: pip install tifffile")
    exit(1)

try:
    from skimage import data, util
except ImportError:
    print("Please install scikit-image: pip install scikit-image")
    exit(1)

out_dir = "../demo_images"
os.makedirs(out_dir, exist_ok=True)
print("Generating standard research test images...\n")

# ─────────────────────────────────────────────────────────
# 1. Low Contrast — Spatial / Histogram Equalization
# ─────────────────────────────────────────────────────────
moon = data.moon()
cv2.imwrite(os.path.join(out_dir, "1_low_contrast_moon.png"), moon)
print(" ✔ 1_low_contrast_moon.png       → Spatial: Histogram Equalization, CLAHE")

# ─────────────────────────────────────────────────────────
# 2. Noisy — Restoration / Morphology / AI Denoise
# ─────────────────────────────────────────────────────────
camera = data.camera()

sp_noisy = util.random_noise(camera, mode='s&p', amount=0.05)
cv2.imwrite(os.path.join(out_dir, "2_noisy_salt_pepper.png"), (sp_noisy * 255).astype(np.uint8))
print(" ✔ 2_noisy_salt_pepper.png       → Morphology: Opening  |  Restoration: Median Filter")

gauss_noisy = util.random_noise(camera, mode='gaussian', var=0.02)
cv2.imwrite(os.path.join(out_dir, "2_noisy_gaussian.png"), (gauss_noisy * 255).astype(np.uint8))
print(" ✔ 2_noisy_gaussian.png          → Deep Learning: AI Denoise  |  Restoration: Wiener")

# ─────────────────────────────────────────────────────────
# 3. Coins — Segmentation / Edge Detection
# ─────────────────────────────────────────────────────────
coins = data.coins()
cv2.imwrite(os.path.join(out_dir, "3_coins_segmentation.png"), coins)
print(" ✔ 3_coins_segmentation.png      → Segmentation: Canny Edge, Otsu Threshold, Watershed")

# ─────────────────────────────────────────────────────────
# 4. Low Resolution — Deep Learning SRCNN
# ─────────────────────────────────────────────────────────
astronaut = data.astronaut()
astronaut_bgr = cv2.cvtColor(astronaut, cv2.COLOR_RGB2BGR)
h, w = astronaut_bgr.shape[:2]
low_res = cv2.resize(astronaut_bgr, (w // 3, h // 3), interpolation=cv2.INTER_CUBIC)
cv2.imwrite(os.path.join(out_dir, "4_low_res_astronaut.png"), low_res)
print(" ✔ 4_low_res_astronaut.png       → Deep Learning: Super Resolution (SRCNN x2)")

# ─────────────────────────────────────────────────────────
# 5. Color — Color Space / False Color / Channel Split
# ─────────────────────────────────────────────────────────
coffee = data.coffee()
coffee_bgr = cv2.cvtColor(coffee, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(out_dir, "5_color_coffee.png"), coffee_bgr)
print(" ✔ 5_color_coffee.png            → Color: False Color (JET), Channel Splitting")

# ─────────────────────────────────────────────────────────
# 6. Synthetic Multispectral TIFF — Remote Sensing (NDVI)
#
#    4-band layout (matching real Sentinel-2 / Landsat convention):
#       Band 0 = Blue        (low reflectance everywhere)
#       Band 1 = Green       (moderate in vegetation)
#       Band 2 = Red         (LOW in vegetation, high in bare soil) ← red_band = 2
#       Band 3 = NIR         (HIGH in vegetation, low in water)     ← nir_band = 3
#
#    Scene regions:
#       Top-left  quadrant = Dense vegetation  → high NDVI (green in colormap)
#       Top-right quadrant = Bare soil/Urban   → near-zero NDVI (yellow)
#       Bottom    half     = Water body        → negative NDVI (red)
# ─────────────────────────────────────────────────────────

H, W = 256, 256
band_blue  = np.zeros((H, W), dtype=np.uint8)
band_green = np.zeros((H, W), dtype=np.uint8)
band_red   = np.zeros((H, W), dtype=np.uint8)
band_nir   = np.zeros((H, W), dtype=np.uint8)

# Region masks
veg_mask  = np.zeros((H, W), bool)
soil_mask = np.zeros((H, W), bool)
water_mask = np.zeros((H, W), bool)

veg_mask[:H//2, :W//2]    = True    # top-left: vegetation
soil_mask[:H//2, W//2:]   = True    # top-right: soil/urban
water_mask[H//2:, :]      = True    # bottom: water

# Vegetation: NIR high (~200), Red low (~40)
band_nir[veg_mask]   = np.random.randint(180, 230, size=veg_mask.sum())
band_red[veg_mask]   = np.random.randint(20,  60,  size=veg_mask.sum())
band_green[veg_mask] = np.random.randint(80,  120, size=veg_mask.sum())
band_blue[veg_mask]  = np.random.randint(20,  50,  size=veg_mask.sum())

# Bare soil: NIR and Red similar (~120-140)
band_nir[soil_mask]   = np.random.randint(110, 150, size=soil_mask.sum())
band_red[soil_mask]   = np.random.randint(100, 140, size=soil_mask.sum())
band_green[soil_mask] = np.random.randint(80,  120, size=soil_mask.sum())
band_blue[soil_mask]  = np.random.randint(60,  100, size=soil_mask.sum())

# Water: NIR very low (~20), Red low (~40) → negative NDVI
band_nir[water_mask]   = np.random.randint(10, 40,  size=water_mask.sum())
band_red[water_mask]   = np.random.randint(30, 70,  size=water_mask.sum())
band_green[water_mask] = np.random.randint(40, 80,  size=water_mask.sum())
band_blue[water_mask]  = np.random.randint(80, 130, size=water_mask.sum())

# Stack to (H, W, 4) — tifffile expects this for multiband
multiband = np.stack([band_blue, band_green, band_red, band_nir], axis=2)

# Save as multi-band TIFF (the format the NDVI endpoint is designed for)
out_tiff = os.path.join(out_dir, "6_multispectral_ndvi.tif")
tifffile.imwrite(out_tiff, multiband)
print(" ✔ 6_multispectral_ndvi.tif      → Remote Sensing: NDVI (nir_band=3, red_band=2)")
print("   Tip: Also try Band Viewer (band 0-3) and Band Statistics on this image.")

print("\n✅ All 7 demo images generated in the 'demo_images/' folder!")
print("\nBand layout of 6_multispectral_ndvi.tif:")
print("  Band 0 = Blue  | Band 1 = Green  | Band 2 = Red  | Band 3 = NIR")
print("  → For NDVI: set nir_band = 3, red_band = 2")
