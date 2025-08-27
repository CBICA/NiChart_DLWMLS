import argparse
import os
import shutil
import logging
import pandas as pd

from .utils import (
    reorient_to_lps,
    run_DLWMLS,
    register_flair_to_t1,
    apply_saved_transform,
    segment_multilabel_mask_and_calculate_volumes
)

VERSION = "0.0.1"


def main() -> None:
    prog = "NiChart_DLWMLS"
    description = "NiCHART White Matter Lesion Segmentation Pipeline"
    usage = """
    NiChart_DLWMLS v{VERSION}
    Run WML Segmentation and secondary segementation using DLMUSE masks

    Required arguments:
        [-fl, --fl_dir] : Name of the input folder with FL scans  (REQUIRED)
        [-o, --out_dir] : Name of the output folder for segmentation (REQUIRED)
        [--list]        List of MRIDs; first raw (column header) skipped (OPTIONAL)
        [--t1_dir]      Name of the input folder with T1 scans  (OPTIONAL)
        [--t1_suff]     Suffix of the input T1 scans (OPTIONAL, DEFAULT: _T1.nii.gz)
        [--dlmuse_dir]  Name of the input folder with T1 scans  (OPTIONAL)
        [--dlmuse_suff] Suffix of the input T1 scans (OPTIONAL, DEFAULT: _T1_LPS_DLMUSE.nii.gz)
    
    Optional arguments:
        [-r, --remove_intermediate]  Remove all intermediate files. (DEFAULT: True)
        [-d, --device]  Device to run segmentation ('cuda' (GPU), 'cpu' (CPU) or 
                        'mps' (Apple M-series chips supporting 3D CNN))
        [-h, --help]    Show this help message and exit.
        [-V, --version] Show program's version number and exit.
        
    EXAMPLE USAGE:

        Executing the full pipeline including seperating WMLS mask into Brain ROI level 
            based on the input DLMUSE masks

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
    """.format(
        VERSION=VERSION
    )

    parser = argparse.ArgumentParser(
        prog=prog, usage=usage, description=description, add_help=False
    )

    parser.add_argument('--fl_dir', required=True, type=str, help='Name of the input folder with FL scans (REQUIRED)')
    parser.add_argument('--fl_suff', type=str, default='_FL.nii.gz', help='Suffix of the input FLAIR scans (OPTIONAL, DEFAULT: _FL.nii.gz)')
    
    parser.add_argument('--out_dir', required=True, type=str, help='Name of the output folder for segmentation (REQUIRED)')
    
    parser.add_argument('--list', type=str, default=None, help='List of MRIDs; first row (column header) skipped (OPTIONAL)')
    
    parser.add_argument('--t1_dir', required=True, type=str, default=None, help='Name of the input folder with T1 scans (OPTIONAL)')
    parser.add_argument('--t1_suff', type=str, default='_T1.nii.gz', help='Suffix of the input T1 scans (OPTIONAL, DEFAULT: _T1.nii.gz)')
    
    parser.add_argument('--dlmuse_dir', required=True, type=str, default=None, help='Name of the input folder with DLMUSE masks (OPTIONAL)')
    parser.add_argument('--dlmuse_suff', type=str, default='_T1_LPS_DLMUSE.nii.gz', help='Suffix of the input DLMUSE masks (OPTIONAL, DEFAULT: _T1_LPS_DLMUSE.nii.gz)')
    parser.add_argument('-r', '--remove_intermediate', type=str, default='True', help="Remove all intermediate files (Default: True)")
    parser.add_argument('-d', '--device', type=str, default="cuda", help="Device to run segmentation ('cuda' (GPU), 'cpu' (CPU) or 'mps' (Apple M-series chips supporting 3D CNN))")
    
    parser.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS, help='Show this help message and exit.')
    
    parser.add_argument('-V', '--version', action='version', version=f'%(prog)s {VERSION}', help="Show program's version number and exit.")

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
    # Suffixes for intermediate files
    dlwmls_suffix = '_FL_LPS_DLWMLS.nii.gz'
    fl_to_t1_xfm_suffix = '_FL_to_T1.tfm'
    dlwmls_to_t1_reg_suffix = '_DLWMLS_REG_to_T1.nii.gz'
    dlwmls_dlmuse_segmented_suffix = "_DLWMLS_DLMUSE_Segmented.nii.gz"
    dlwmls_roi_volume_csv_suffix = '_DLWMLS_DLMUSE_Segmented_Volumes.csv'

    # Other args
    remove_intermediate = args.remove_intermediate.lower() == 'true'
    
    if not os.path.exists(output_directory):
        logging.warning(f"Output folder '{output_directory}' not found. Creating '{output_directory}'")
        os.mkdir(output_directory)
    else:
        shutil.rmtree(output_directory)
        logging.warning(f"Output folder '{output_directory}' found. Removing existing files and re-creating '{output_directory}'")
        os.mkdir(output_directory)

    flair_lps_path = os.path.join(output_directory, 'FLAIR_LPS')
    t1_lps_path = os.path.join(output_directory, 'T1_LPS')
    dlwmls_path = os.path.join(output_directory, 'DLWMLS')
    tfm_path = os.path.join(output_directory,'TFMs')
    dlwmls_tfmed = os.path.join(output_directory,'DLWMLS_TFM_to_T1')
    dlwmls_dlmuse_segmented_path = os.path.join(output_directory,'DLWMLS_DLMUSE_Segmented')
    
    os.mkdir(flair_lps_path)
    os.mkdir(t1_lps_path)
    os.mkdir(dlwmls_path)
    os.mkdir(tfm_path)
    os.mkdir(dlwmls_tfmed)
    os.mkdir(dlwmls_dlmuse_segmented_path)

    df_list = pd.read_csv(args.list)
    mrids = df_list.iloc[:, 0].tolist()

    #####################################################
    ########## START NiChart_DLWMLS Pipeline ############
    #####################################################

    logging.info(f"LPS Orienting and saving the images")
    for mrid in mrids:
        # Reorient T1
        reorient_to_lps(input_path=os.path.join(t1_path, mrid + t1_image_suffix),
                        output_path=os.path.join(t1_lps_path, mrid + t1_image_suffix))
        # Reorient FLAIR
        reorient_to_lps(input_path=os.path.join(fl_path, mrid + fl_image_suffix),
                        output_path=os.path.join(flair_lps_path, mrid + fl_image_suffix))
        
    logging.info(f"Processing DLWMLS on FLAIR folder")
    # Check if the folder exists
    
    run_DLWMLS(in_dir=flair_lps_path, 
               out_dir=dlwmls_path,
               device=args.device)

    
    logging.info(f"Creating transformation matrix from FL to T1, applying to the DLWMLS Masks")
    for mrid in mrids:
        try:
            register_flair_to_t1(t1_image_path=os.path.join(t1_lps_path, mrid + t1_image_suffix),
                                flair_image_path=os.path.join(flair_lps_path, mrid + fl_image_suffix),
                                output_path=os.path.join(tfm_path, mrid+fl_to_t1_xfm_suffix))
            
            apply_saved_transform(fixed_image_path=os.path.join(t1_lps_path, mrid + t1_image_suffix),
                                moving_image_path=os.path.join(dlwmls_path, mrid + dlwmls_suffix),
                                transform_path=os.path.join(tfm_path, mrid + fl_to_t1_xfm_suffix),
                                output_image_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix))
            
            segment_multilabel_mask_and_calculate_volumes(mask_a_path=os.path.join(dlwmls_tfmed, mrid + dlwmls_to_t1_reg_suffix),
                                                        mask_b_path=os.path.join(dlmuse_directory, mrid + dlmuse_suffix),
                                                        output_path=os.path.join(dlwmls_dlmuse_segmented_path, mrid + dlwmls_dlmuse_segmented_suffix),
                                                        save_as_csv=True,
                                                        csv_path=os.path.join(output_directory, mrid + dlwmls_roi_volume_csv_suffix),
                                                        mrid = mrid)
        except Exception as e:
            print(f"{mrid} excluded due to {e}")

    if remove_intermediate:
        shutil.rmtree(flair_lps_path)
        shutil.rmtree(t1_lps_path)
        shutil.rmtree(dlwmls_path)
        shutil.rmtree(tfm_path)
        shutil.rmtree(dlwmls_tfmed)

if __name__ == "__main__":
    #print("Please use CMD to run NiChart_DLWMLS or NiChart_DLWMLS_essential.")
    main()