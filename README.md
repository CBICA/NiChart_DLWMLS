# NiChart_DLWMLS

Run Deep-Learning-based-White-Matter-Lesion-Segmentation on your data (requires FLAIR, optional T1 masks for granular segmentation).

Executing the full pipeline including seperating WMLS mask into Brain ROI level based on the input DLMUSE masks.

## Installation

#### 0. Create a conda env (Python >= 3.9)

#### 1. Install DLWMLS (Required dependency)
```bash
git clone https://github.com/CBICA/DLWMLS.git
cd DLWMLS
pip install -e .
```

### 1. Install wmh_seg (Optional) 
```bash
git clone https://github.com/euroso97/wmh_seg.git
cd wmh_seg
wget https://huggingface.co/jil202/wmh_seg/resolve/main/ChallengeMatched_Unet_mit_b5.pth

```

#### 2. Install NiChart_DLWMLS
```bash
git clone https://github.com/CBICA/NiChart_DLWMLS.git
cd NiChart_DLWMLS
pip install -e .
```

## Usage

### Prerequisits:
- T1 image (.nii.gz)
- FL image (.nii.gz)
- DLMUSE mask (.nii.gz) (refer to: [NiChart_DLMUSE](https://github.com/CBICA/NiChart_DLMUSE))
- (*optional) your own WMH mask (.nii.gz)

#### Required arguments:

    [-fl, --fl_dir] : Name of the input folder with FL scans  (REQUIRED)
    [-o, --out_dir] : Name of the output folder for segmentation (REQUIRED)
    [--list]        List of MRIDs; first raw (column header) skipped (OPTIONAL)
    [--t1_dir]      Name of the input folder with T1 scans  (OPTIONAL)
    [--t1_suff]     Suffix of the input T1 scans (OPTIONAL, DEFAULT: _T1.nii.gz)
    [--dlmuse_dir]  Name of the input folder with T1 scans  (OPTIONAL)
    [--dlmuse_suff] Suffix of the input T1 scans (OPTIONAL, DEFAULT: _T1_LPS_DLMUSE.nii.gz)

#### Optional arguments:

    [-d, --device]  Device to run segmentation ('cuda' (GPU), 'cpu' (CPU) or 
                    'mps' (Apple M-series chips supporting 3D CNN))
    [-h, --help]    Show this help message and exit.
    [-V, --version] Show program's version number and exit.
    
#### EXAMPLE USAGE:

    Executing the full pipeline including seperating WMLS mask into Brain ROI level 
        based on the input DLMUSE masks:

    NiChart_DLWMLS  --list          /path/to/mrid_list.csv \
                    --fl_dir        /path/to/flair_images  \
                    --fl_suff       _FL_LPS.nii.gz         \
                    --t1_dir        /path/to/t1_images     \
                    --t1_suff       _T1_LPS.nii.gz         \
                    --dlmuse_dir    /path/to/dlmuse_masks  \
                    --dlmuse_suff   _T1_LPS_DLMUSE.nii.gz  \
                    --out_dir       /path/to/output


    Using your own WMH masks (skipping DLWMLS segmentation):

        NiChart_DLWMLS  --list          /path/to/mrid_list.csv \
                        --fl_dir        /path/to/flair_images  \
                        --fl_suff       _FL.nii.gz             \
                        --wmh_dir       /path/to/dlwmls_masks  \
                        --wmh_suff      _FL_LPS_DLWMLS.nii.gz  \
                        --t1_dir        /path/to/t1_images     \
                        --t1_suff       _T1.nii.gz             \
                        --dlmuse_dir    /path/to/dlmuse_masks  \
                        --dlmuse_suff   _T1_LPS_DLMUSE.nii.gz  \
                        --out_dir       /path/to/output        \
                        --remove_intermediate True             \
                        --device cpu/cuda
