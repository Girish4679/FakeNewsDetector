# Setup Guide

There are two roles:

| Role | Machine | Job |
|---|---|---|
| **Downloader** | Mac | Downloads images from Reddit into shared Google Drive |
| **Trainer** | Windows + GPU | Reads images from Google Drive, runs training |

---

## Mac Setup (Downloader)

### 1. Find your Google Drive local path

Install **Google Drive for Desktop** and sign in to the shared account.  
On Mac, the path will look like one of these depending on your version:

```
/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/
```

To find it exactly, open Finder → click Google Drive in the sidebar → right-click the address bar → Copy as Pathname.  
Or run this in terminal:

```bash
ls ~/Library/CloudStorage/
# Look for the folder starting with GoogleDrive-
```

You'll use this path in every command below. Replace the example path with your actual one.

---

### 2. Clone the repo

```bash
git clone <repo-url>
cd FakeNewsDetector
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Confirm TSV files are in Google Drive

The TSV files should already be somewhere in your Google Drive. Confirm they exist:

```bash
ls "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/data/"
# Should show: multimodal_train.tsv  multimodal_validate.tsv  multimodal_test_public.tsv
```

---

### 5. Download images

**Start with a test run of 5,000 images to make sure everything works:**

```bash
python image_downloader.py \
  --split train \
  --data_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/data" \
  --output_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/images" \
  --max_images 5000
```

**Once confirmed working, run the full download (stops at 50 GB):**

```bash
python image_downloader.py \
  --split train \
  --data_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/data" \
  --output_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/images" \
  --max_gb 50
```

**Also grab the validate split:**

```bash
python image_downloader.py \
  --split validate \
  --data_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/data" \
  --output_dir "/Users/YourName/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/fakeddit/images" \
  --max_gb 50
```

> Add `--workers 32` to any command above to download faster.

The script is fully resumable — safe to Ctrl+C and re-run. Already-downloaded images are skipped automatically via a `downloaded_ids.txt` manifest file in the images folder.

---

## Windows Setup (Trainer — GPU machine)

### 1. Find your Google Drive local path

Install **Google Drive for Desktop** and sign in to the **same Google account** as the Mac downloader.  
Open File Explorer, find Google Drive in the sidebar, click into the fakeddit folder, and copy the path from the address bar. It will look like:

```
G:\My Drive\fakeddit\
```

---

### 2. Clone the repo

```bash
git clone <repo-url>
cd FakeNewsDetector
```

---

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch with CUDA — go to https://pytorch.org/get-started/locally/ and select your CUDA version. The command will look like:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

### 4. Verify GPU is available

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Should print `True` and your GPU name.

---

### 5. Verify images are synced from Drive

```bash
dir "G:\My Drive\fakeddit\images"
# Should show .jpg files and downloaded_ids.txt
```

Wait for Google Drive to finish syncing before starting training.

---

## For other team members (no GPU, no images)

Generate a small local subset for development — no images needed:

```bash
python scripts/make_subset.py --n 5000
```

This writes `data/subset_train.tsv`, `data/subset_validate.tsv`, `data/subset_test.tsv` with 5,000 balanced samples. Enough to write and test model code locally.
