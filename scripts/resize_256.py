"""Pre-resize the Zero-DCE training images to 256x256 (lossless PNG bytes, original .jpg names kept so the
official loader's glob still matches). Makes the ablation runs ~2.5x faster; the main run used the originals."""
import glob, os, sys
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for p in sorted(glob.glob(f'{src}/*.jpg')):
    Image.open(p).convert('RGB').resize((256, 256), Image.LANCZOS).save(f'{dst}/{os.path.basename(p)}', format='PNG')
