## VPR Baselines

Minimal, single-machine experiment to benchmark global-image-embedding baselines for Visual Place Recognition on a custom dataset. Uses Transformers, timm, and FAISS with a simple Recall\@k evaluation based on GPS thresholds.

### Features

* 5 to 6 baseline encoders: CLIP, OpenCLIP, SigLIP, DINOv2, ConvNeXt-B, ResNet50
* One script to embed, index, search, and evaluate
* CSV-based dataset with absolute image resolution preserved at load
* Exact FAISS search and Recall\@1,5,10 at configurable distance threshold

---

## 1. Setup

```bash
# Create environment
conda create -n vpr-poc python=3.10 -y
conda activate vpr-poc

# PyTorch with CUDA 12.1 (works with 560.xx drivers)
conda install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y

# POC dependencies
pip install transformers==4.42.4 timm==1.0.7 faiss-cpu==1.7.4 \
            pandas==2.2.2 numpy==1.26.4 pillow==10.3.0 accelerate==0.33.0
```

---

## 2. Repository layout

```
vpr-baselines-poc/
  vpr_poc.py            # main script: embeddings + FAISS + eval
  README.md
```

---

## 3. Dataset format

CSV header:

```
camera_id,img_relpath,t_nsec,lat,lon,alt
```

Example row:

```
front_left_center,images/front_left_center/1641321200080306000.png,1641321200080306000,36.274790411,-115.012062955,597.459
```

The script resolves image paths by going two parents up from the CSV path to reach the sequence root, then joining `img_relpath`.

---

## 4. Run baseline models

Train CSV:

```
/media/pragyan/Data/racecar_ws/output/sequences/M-SOLO-SLOW-70-100/poses/poses.csv
```

Test CSV:

```
/media/pragyan/Data/racecar_ws/output/sequences/M-MULTI-SLOW-KAIST/poses/poses.csv
```

Pick your camera view, for example `front_left_center`. Input size 224 is the default for this POC.

```bash
# CLIP ViT-B/32
python vpr_poc.py \
  --train_csv /media/pragyan/Data/racecar_ws/output/sequences/M-SOLO-SLOW-70-100/poses/poses.csv \
  --test_csv  /media/pragyan/Data/racecar_ws/output/sequences/M-MULTI-SLOW-KAIST/poses/poses.csv \
  --camera_id front_left_center --model clip_b32 --input 224 --batch 128 --pos_m 10 --k 10 \
  --save_npz runs_clip_b32_224

# CLIP ViT-L/14
python vpr_poc.py --train_csv ... --test_csv ... \
  --camera_id front_left_center --model clip_l14 --input 224 --batch 32 --pos_m 10 --k 10 \
  --save_npz runs_clip_l14_224

# SigLIP base-p16-224
python vpr_poc.py --train_csv ... --test_csv ... \
  --camera_id front_left_center --model siglip_b16 --input 224 --batch 128 --pos_m 10 --k 10 \
  --save_npz runs_siglip_b16_224

# DINOv2 ViT-B
python vpr_poc.py --train_csv ... --test_csv ... \
  --camera_id front_left_center --model dinov2_b --input 224 --batch 128 --pos_m 10 --k 10 \
  --save_npz runs_dinov2_b_224

# ConvNeXt-B
python vpr_poc.py --train_csv ... --test_csv ... \
  --camera_id front_left_center --model convnext_b --input 224 --batch 256 --pos_m 10 --k 10 \
  --save_npz runs_convnext_b_224

# ResNet50
python vpr_poc.py --train_csv ... --test_csv ... \
  --camera_id front_left_center --model resnet50 --input 224 --batch 256 --pos_m 10 --k 10 \
  --save_npz runs_resnet50_224
```

---

## 5. Output

* Prints Recall\@1, Recall\@5, Recall\@10 on queries that have at least one positive within `--pos_m` meters
* Saves compressed results when `--save_npz` is set:

  * Train and test embeddings in float16
  * FAISS top-k indices and similarities
  * Lat, lon arrays for reproducible analysis

---

## 6. Evaluation definition

* A test image is considered valid if there exists at least one train image within `--pos_m` meters by haversine distance
* Recall\@k is computed over the valid subset
* Default threshold is 10 m, configurable via `--pos_m`

---

## 7. Practical tips

* Your RTX 4090 should handle the suggested batch sizes. Lower batches if you see OOM
* Keep the POC simple at 224 input. Later you can add 336 or multi-crop averaging across width
* CPU FAISS is enough at this scale. Upgrade to FAISS-GPU later if needed

---

## License

MIT

---

## Acknowledgments

Built with PyTorch, Hugging Face Transformers, and timm.
