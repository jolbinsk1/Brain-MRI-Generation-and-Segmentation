import os
from PIL import Image

images_dir = '/home/u893875/nnUNet/nnUNet_raw_2/Dataset102_UCSF_PDGM_pix2pix_full/imagesTs'

files = sorted([f for f in os.listdir(images_dir) if f.endswith('_0002.png')])
print(f"Found {len(files)} files to convert.\n")

for fname in files:
    fpath = os.path.join(images_dir, fname)
    img = Image.open(fpath).convert('L')  # L = single channel grayscale
    img.save(fpath)
    print(f"Converted: {fname}")

print("\nDone.")
