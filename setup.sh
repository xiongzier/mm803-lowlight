#!/usr/bin/env bash
# Clone the two upstream repos at the commits we used, apply our small fixes, and download data + official weights.
set -e
cd "$(dirname "$0")"
git clone https://github.com/Li-Chongyi/Zero-DCE.git
git -C Zero-DCE checkout e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd
git -C Zero-DCE apply ../patches/zero-dce.patch
cp scripts/zerodce_train_ablation.py Zero-DCE/Zero-DCE_code/lowlight_train_ablation.py
mv Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth Zero-DCE/official_Epoch99.pth

git clone https://github.com/caiyuanhao1998/Retinexformer.git
git -C Retinexformer checkout 2dd9ae05f34a34e2df3b955f743215ac3cda0f82
cp configs/Repro_RetinexFormer_LOL_v1.yml Retinexformer/Options/
ln -sfn ../data Retinexformer/data

mkdir -p data/raw data/unpaired pretrained/retinexformer
gdown 1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H -O data/raw/LOLv1.zip            # LOL-v1 (Retinexformer repo link)
gdown 1GAB3uGsmAyLgtDBDONbil08vVu5wJcG3 -O data/raw/zerodce_train.zip    # Zero-DCE training set (SICE part 1)
gdown --folder 1RR50EJYGIHaUYwq4NtK7dx8faMSvX8Xp -O data/unpaired       # DICM / LIME / MEF / NPE / VV
gdown --folder 1ynK5hfQachzc8y96ZumhkPPDXzHJwaQV -O pretrained/retinexformer   # official Retinexformer weights
unzip -q data/raw/LOLv1.zip -d data/
unzip -q data/raw/zerodce_train.zip -d data/zerodce_train/
echo "Done. Check that data/LOLv1/{Train,Test}/{input,target} exist (move up one level if the zip nests LOLv1/LOLv1)."
