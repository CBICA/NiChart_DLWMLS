import argparse
import glob
import os
import sys
import shutil
import logging
import pandas as pd
from pathlib import Path

from .utils import (
    reorient_to_lps,
    run_DLWMLS,
    register_flair_to_t1,
    apply_saved_transform,
    segment_multilabel_mask_and_calculate_volumes,
    strip_filename,
    fill_missing_paths,
)

from NiChart_common_utils.nifti_parser import NiftiMRIDParser

VERSION = "0.1.0"


def main() -> None:
    prog = "NiChart_DLWMLS"
    description = "NiCHART White Matter Hyperintensity Segmentation Pipeline (Brain ROI level)"
    usage = """
    NiChart_DLWMLS v{VERSION}
    Run WMH Segmentation and secondary segementation using ROI masks.
    Optionally, use your own WMH mask to segment it into ROI level.

    Required arguments:
        [--list]        Path to a CSV row list of MRIDs with a single column name MRID 
                        (Your files must be named MRID + Suffix, or otherwise have T1, FLAIR, DLMUSE headers in list csv. See --infer-mrids for alternative.)
        [-fl, --fl_dir] Name of the input folder with FL scans
        [--fl_suff]     Suffix of the input FLAIR scans (DEFAULT: _FL.nii.gz)
        [--t1_dir]      Name of the input folder with T1 scans
        [--t1_suff]     Suffix of the input T1 scans (DEFAULT: _T1.nii.gz)
        [--dlmuse_dir]  Name of the input folder with DLMUSE masks
        [--dlmuse_suff] Suffix of the input DLMUSE masks (DEFAULT: _T1_LPS_DLMUSE.nii.gz)
        [-o, --out_dir] Name of the output folder for segmentation outputs
    
    Optional arguments:
        [--wmh_dir]  Name of the input folder with White Matter Hyperintensity masks
                        The masks should match the orientation & dimension of the input FL images.
                        Entering this will bypass the initial WMH segmentation step.
        [--wmh_suff] Suffix of the input White Matter Hyperintensity masks (DEFAULT: _FL_LPS_DLWMLS.nii.gz)
        [-r, --remove_intermediate]  Remove all intermediate files. (DEFAULT: True)
        [--infer-mrids] Use heuristic NIFTI filename parsing instead of strict suffix requirement (experimental)
        [-d, --device]  Device to run segmentation ('cuda' (GPU), 'cpu' (CPU) or 
                        'mps' (Apple M-series chips supporting 3D CNN))
        [--named-headers] Change column headers to format DL_WMLS_Volume_# instead of raw integers
        [-h, --help]    Show this help message and exit.
        [-V, --version] Show program's version number and exit.
        
    EXAMPLE USAGE:

        Executing the full pipeline including seperating WMLS mask into Brain ROI level 
            based on the input DLMUSE masks:

        NiChart_DLWMLS  --list          /path/to/mrid_list.csv \
                        --fl_dir        /path/to/flair_images  \
                        --fl_suff       _FL.nii.gz             \
                        --t1_dir        /path/to/t1_images     \
                        --t1_suff       _T1.nii.gz             \
                        --dlmuse_dir    /path/to/dlmuse_masks  \
                        --dlmuse_suff   _T1_LPS_DLMUSE.nii.gz  \
                        --out_dir       /path/to/output        \
                        --remove_intermediate True             \
                        --device cpu/cuda

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
    """.format(
        VERSION=VERSION
    )

    parser = argparse.ArgumentParser(
        prog=prog, usage=usage, description=description, add_help=False
    )

    parser.add_argument('--list', type=str, default=None)

    parser.add_argument('--fl_dir', required=True, type=str)
    parser.add_argument('--fl_suff', type=str, default='_FL.nii.gz') 
    
    parser.add_argument('--t1_dir', required=True, type=str, default=None)
    parser.add_argument('--t1_suff', type=str, default='_T1.nii.gz')
    
    parser.add_argument('--wmh_dir', type=str, default='') # Optional
    parser.add_argument('--wmh_suff', type=str, default='_FL_LPS_DLMUSE.nii.gz')

    parser.add_argument('--dlmuse_dir', required=True, type=str, default=None)
    parser.add_argument('--dlmuse_suff', type=str, default='_T1_LPS_DLMUSE.nii.gz')
    
    parser.add_argument('--out_dir', required=True, type=str)

    parser.add_argument('-r', '--remove_intermediate', type=str, default='True')
    parser.add_argument('--named-headers', action="store_true", default=False)
    parser.add_argument('--infer-mrids', action="store_true", default=False)
    parser.add_argument('-d', '--device', type=str, default="cuda")
    
    # parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS)
    
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}')

    args = parser.parse_args()

    # For demonstration, print the parsed arguments (remove or replace with pipeline logic as needed)
    print('Parsed arguments:')
    for arg, value in vars(args).items():
        print(f'  {arg}: {value}')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # Suffixes to verify input files
    t1_image_suffix = args.t1_suff
    fl_image_suffix = args.fl_suff
    t1_path = args.t1_dir
    fl_path = args.fl_dir

    output_directory = args.out_dir
    dlmuse_directory = args.dlmuse_dir
    dlmuse_suffix = args.dlmuse_suff
    user_wmh_directory = args.wmh_dir
    user_wmh_suffix = args.wmh_suff
    # Suffixes for intermediate files
    t1_lps_suffix = '_T1_LPS.nii.gz'
    fl_lps_suffix = '_FL_LPS.nii.gz'
    dlwmls_suffix = '_FL_LPS_DLWMLS.nii.gz'
    fl_to_t1_xfm_suffix = '_FL_to_T1.tfm'
    dlwmls_to_t1_reg_suffix = '_DLWMLS_REG_to_T1.nii.gz'
    dlwmls_dlmuse_segmented_suffix = "_DLWMLS_DLMUSE_Segmented.nii.gz"
    dlwmls_roi_volume_csv_suffix = '_DLWMLS_DLMUSE_Segmented_Volumes.csv'

    # Other args
    remove_intermediate = args.remove_intermediate.lower() == 'true'
    
    if not os.path.exists(output_directory):
        logging.warning(f"Output folder '{output_directory}' not found. Creating '{output_directory}'")
        os.makedirs(output_directory)

    flair_lps_path = os.path.join(output_directory, 'FLAIR_LPS')
    t1_lps_path = os.path.join(output_directory, 'T1_LPS')
    dlwmls_path = os.path.join(output_directory, 'DLWMLS')
    tfm_path = os.path.join(output_directory,'TFMs')
    dlwmls_tfmed = os.path.join(output_directory,'DLWMLS_TFM_to_T1')
    dlwmls_dlmuse_segmented_path = os.path.join(output_directory,'DLWMLS_DLMUSE_Segmented')
    
    os.makedirs(flair_lps_path, exist_ok=True)
    os.makedirs(t1_lps_path, exist_ok=True)
    os.makedirs(dlwmls_path, exist_ok=True)
    os.makedirs(tfm_path, exist_ok=True)
    os.makedirs(dlwmls_tfmed, exist_ok=True)
    os.makedirs(dlwmls_dlmuse_segmented_path, exist_ok=True)
    
    # Handle super fun subject list cases
    if args.list is not None:
        if args.infer_mrids:
            df_list_original = pd.read_csv(args.list)
            original_mrids = df_list_original.iloc[:, 0].tolist()
            print(f"""
                 --infer-mrids was passed. We'll try heuristic filename parsing on the input and place the inferred index in the output as inferred_data_index.csv.
                 """)
            nifti_parser = NiftiMRIDParser()
            required = ['T1', 'FLAIR', 'DLMUSE']
            if args.wmh_dir:
                required.append('WMH')
            raw_and_derived_dirs = {'T1': t1_path, 'FLAIR': fl_path, 'DLMUSE': dlmuse_directory, 'WMH': user_wmh_directory}
            heuristic_df = nifti_parser.create_master_csv(raw_and_derived_dirs, os.path.join(output_directory, 'inferred_data_index_paths.csv'))
            fill_fn = lambda mrid, req: os.path.basename(nifti_parser.get_path(mrid, req))
             
            df_filled = fill_missing_paths(df_list_original, required, fill_fn)
            print("Final processed subject list:")
            print(df_filled)
            df_filled.to_csv(os.path.join(output_directory, 'inferred_data_index.csv'), index=False)
            df_list = df_filled
            mrids = [str(m) for m in df_list.iloc[:, 0].tolist()]
        else:
            df_list = pd.read_csv(args.list)
            mrids = [str(m) for m in df_list.iloc[:, 0].tolist()]
    else: # default behavior for no provided list csv
        print("""
             No list csv was provided. Please provide a list csv with a single column header, MRID, with each row being an MRID to process.
             By default, we look for the format {MRID}_{suffix} with suffixes specified by the arguments --dlmuse_suff, --fl_suff, --t1_suff, --wmh_suff.
             If your files do not match the format {MRID}_{suffix}, please provide these in additional columns in the list CSV. MRID must be first.
             Example header:
               MRID,T1,FLAIR,WMH,DLMUSE
             Entries under the T1,FLAIR,WMH,DLMUSE columns may be filenames or paths.
             The corresponding directory arguments (--t1_dir, --fl_dir, etc) are still required and the files must still be in these directories.
             Pass --infer-mrids to use experimental heuristic-based filename parsing.
             """)
        sys.exit(1)


    #####################################################
    ########## START NiChart_DLWMLS Pipeline ############
    #####################################################

    logging.info(f"LPS Orienting and saving the images")
    for mrid in mrids:
        # Identify paths in list CSV, if available
        row = df_list[df_list.iloc[:, 0] == mrid]
        mrid_t1_path = None
        mrid_flair_path = None
        if not row.empty:
            if "T1" in row.columns:
                mrid_t1_path = row["T1"].values[0]
            if "FLAIR" in row.columns:
                mrid_flair_path = row["FLAIR"].values[0]
        if not mrid_t1_path:
            candidate = mrid + t1_image_suffix
            mrid_t1_path = candidate
            if not os.path.exists(candidate):
                print(f"WARNING: candidate MRID {mrid} T1 path {candidate} does not exist. This subject may fail. Please specify FLAIR in list CSV or pass --infer-mrids.")
        if not mrid_flair_path:
            candidate = mrid + fl_image_suffix
            mrid_flair_path = candidate
            if not os.path.exists(candidate):
                print(f"WARNING: candidate MRID {mrid} FLAIR path {candidate} does not exist. This subject may fail. Please specify FLAIR in list CSV or pass --infer-mrids.")
           
        try:
            # Reorient T1
            reorient_to_lps(input_path=os.path.join(t1_path, mrid_t1_path),
                            output_path=os.path.join(t1_lps_path, mrid + t1_lps_suffix))
            # Reorient FLAIR
            reorient_to_lps(input_path=os.path.join(fl_path, mrid_flair_path),
                            output_path=os.path.join(flair_lps_path, mrid + fl_lps_suffix))
        except Exception as e:
            logging.info(f"{mrid} T1 or FL LPS orientation failed. Log: {e}")
        
    
    if str(user_wmh_directory) == "":
        use_dlwmls = True
        logging.info(f"Processing DLWMLS on FLAIR folder")
        run_DLWMLS(in_dir=flair_lps_path, 
                out_dir=dlwmls_path,
                device=args.device)
    else:
        # Check if directory with user wmh mask exists
        if os.path.isdir(user_wmh_directory):
            use_dlwmls = False
            logging.info(f"Skipping DLWMLS...using user input WMH masks")
            for mrid in mrids:
                row = df_list[df_list.iloc[:, 0] == mrid]
                mrid_wmh_path = None
                if not row.empty:
                    if "WMH" in row.columns:
                        mrid_wmh_path = row["WMH"].values[0]
                if not mrid_wmh_path:
                    candidate = mrid + user_wmh_suffix
                    mrid_wmh_path = candidate
                    if not os.path.exists(candidate):
                        print(f"WARNING: Candidate MRID {mrid} WMH path {candidate} does not exist. This subject may fail. Please provide WMH filename in list csv or pass --infer-mrids.") 
                try:
                # Reorient WMH mask
                    reorient_to_lps(input_path=os.path.join(user_wmh_directory, mrid_wmh_path),
                                    output_path=os.path.join(dlwmls_path, mrid + dlwmls_suffix))
                except Exception as e:
                    logging.info(f"{mrid} WMH mask LPS orientation failed. Log: {e}")
        else:
            logging.warning(f"Invalid user input WMH path")

    
    logging.info(f"Creating transformation matrix from FL to T1, applying to the DLWMLS Masks")
    for mrid in mrids:
        row = df_list[df_list.iloc[:, 0] == mrid]
        mrid_dlmuse_path = None
        
        if not row.empty:
            if "DLMUSE" in row.columns:
                mrid_dlmuse_path = row["DLMUSE"].values[0]
        if not mrid_dlmuse_path:
            candidate = mrid + dlmuse_suffix
            mrid_dlmuse_path = candidate
            if not os.path.exists(candidate):
                print(f"WARNING: Candidate MRID {mrid} DLMUSE path {candidate} does not exist. This subject may fail. Please provide DLMUSE filename in list csv or pass --infer-mrids.") 
        
        try:
            register_flair_to_t1(t1_image_path=os.path.join(t1_lps_path, mrid + t1_lps_suffix),
                                flair_image_path=os.path.join(flair_lps_path, mrid + fl_lps_suffix),
                                output_path=os.path.join(tfm_path, mrid+fl_to_t1_xfm_suffix))
            
            apply_saved_transform(fixed_image_path=os.path.join(t1_lps_path, mrid + t1_lps_suffix),
                                moving_image_path=os.path.join(dlwmls_path, mrid + dlwmls_suffix),
                                transform_path=os.path.join(tfm_path, mrid + fl_to_t1_xfm_suffix),
                                output_image_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix))
            # Copy this subject's FLAIR-space lesion segmentation to output
            if use_dlwmls:
                dlwmls_outputcopy_path = os.path.join(dlwmls_path, mrid + dlwmls_suffix)
                dlwmls_flair_mask_subdir = os.path.join(output_directory, "DLWMLS_FLAIR")
                dlwmls_mask_output_path = os.path.join(dlwmls_flair_mask_subdir, mrid + dlwmls_suffix)
                print(f"Copying FLAIR-space segmentation to output location {dlwmls_mask_output_path}")
                os.makedirs(dlwmls_flair_mask_subdir, exist_ok=True)
                shutil.copy(dlwmls_outputcopy_path, dlwmls_mask_output_path)

            segment_multilabel_mask_and_calculate_volumes(mask_a_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix),
                                                        mask_b_path=os.path.join(dlmuse_directory, mrid_dlmuse_path),
                                                        output_path=os.path.join(dlwmls_dlmuse_segmented_path, mrid + dlwmls_dlmuse_segmented_suffix),
                                                        save_as_csv=True,
                                                        csv_path=os.path.join(output_directory, mrid + dlwmls_roi_volume_csv_suffix),
                                                        mrid = mrid)
        except Exception as e:
            print(f"{mrid} excluded due to {e}")
    
    pattern = os.path.join(output_directory, "*" + dlwmls_roi_volume_csv_suffix)
    csv_files = glob.glob(pattern)
    df_all_csvs = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
    
    
    if args.named_headers:
        def rename_col(col):
            if str(col).isdigit():
                return f"DL_WMLS_Volume_{col}"
            return col
        df_all_csvs = df_all_csvs.rename(columns={col: rename_col(col) for col in df_all_csvs.columns})

    df_all_csvs.to_csv(os.path.join(output_directory, "DLWMLS_DLMUSE_Segmented_Volumes.csv"), index=False)
    print(f"Merged {len(csv_files)} files into DLWMLS_DLMUSE_Segmented_Volumes.csv")

    if remove_intermediate:
        shutil.rmtree(flair_lps_path)
        shutil.rmtree(t1_lps_path)
        shutil.rmtree(dlwmls_path)
        shutil.rmtree(tfm_path)
        shutil.rmtree(dlwmls_tfmed)

if __name__ == "__main__":
    #print("Please use CMD to run NiChart_DLWMLS or NiChart_DLWMLS_essential.")
    main()
