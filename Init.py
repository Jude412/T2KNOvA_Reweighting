"""This code initializes the analysis by importing the files to reweight, creating the 3D and 8D samples, and computing the bootstrapped
SWD distribution for the target_test 3D and 8D samples. The sampels are saved in the "saved_samples + --output_dir" directory, and the SWD distribution is
saved in the "saved_swd_distribution + --output_dir" directory."""

# Imports 
from NEUTFile_conv_ndim import convert_NEUT_input_file_ndim
from GENIEFile_conv_ndim import convert_GENIE_input_file_ndim
from Sample_creation import create_samples
import numpy as np
import argparse
import os

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Script to create the 3D and 8D training and validation samples and bootstrappedSWD distribution for the reweighting techniques.")
    argparser.add_argument("--input_file_NEUT", required=False, type=str, help="Path to the input ROOT file from Nuisance (that used NEUT as generator).",
                           default="/home/hep/tlt26/RW_Snakemake/NEUT_file/T2KND_FHC_numu_H2O_NEUT562_1M_0000_NUISFLAT.root")
    argparser.add_argument("--input_file_GENIE", required=False, type=str, help="Path to the input ROOT file from Nuisance (that used GENIE as generator).",
                           default="/home/hep/tlt26/RW_Snakemake/GENIE_file/T2KND_FHC_numu_H2O_GENIEv3_G18_10b_00_000_1M_0000_NUISFLAT.root")
    argparser.add_argument("--mode", type=int, help="Interaction mode to select (e.g. 1 for CCQE).",
                           required=False, default=1)
    argparser.add_argument("--neutrino_PDG", type=int, help="PDG code of the neutrino type to select (e.g. 14 for numu).", 
                           required=False, default = 14)
    argparser.add_argument("--train_percentage", type=float, help="Percentage of the training sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--val_percentage", type=float, help="Percentage of the validation sample (between 0 and 1).",
                           required=False, default=0.4)
    argparser.add_argument("--random_seeds", type = list, required=False, 
                        help="List of 2 random seeds to use for the training and validation split.",default= [42, 43])
    argparser.add_argument("--output_dir_samples_3D", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/home/hep/tlt26/RW_Snakemake/saved_samples/first_test/3D/")
    argparser.add_argument("--output_dir_samples_8D", required=False, type=str, help="Path to the output directory where the splitted samples will be saved.",
                            default="/home/hep/tlt26/RW_Snakemake/saved_samples/first_test/8D/")
    args = argparser.parse_args()

    # Getting data from the files
    print("Getting data from the files...")
    List_parameters = ["Enu_true", "ELep", "CosLep", "Q2", "q0", "q3", "W", "Eav"]
    original_8D = convert_NEUT_input_file_ndim(args.input_file_NEUT, mode = args.mode, neutrino_PDG = args.neutrino_PDG, list_parameters = List_parameters)
    target_8D = convert_GENIE_input_file_ndim(args.input_file_GENIE, mode = args.mode, neutrino_PDG = args.neutrino_PDG, list_parameters = List_parameters)

    original_3D = original_8D[:, :3]
    target_3D = target_8D[:, :3]

    # Splitting data
    print("Splitting data into training, validation and test samples...")
    original_8D_train, original_8D_val, original_8D_test = create_samples(original_8D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_8D_train, target_8D_val, target_8D_test = create_samples(target_8D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    original_3D_train, original_3D_val, original_3D_test = create_samples(original_3D, args.train_percentage, args.val_percentage, args.random_seeds[0])
    target_3D_train, target_3D_val, target_3D_test = create_samples(target_3D, args.train_percentage, args.val_percentage, args.random_seeds[1])

    # We save the splitted samples as csv files
    os.makedirs(os.path.join(args.output_dir_samples_3D), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir_samples_8D), exist_ok=True)

    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_train.csv"), original_8D_train, delimiter=",", header=",".join(List_parameters))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_val.csv"), original_8D_val, delimiter=",", header=",".join(List_parameters))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "original_test.csv"), original_8D_test, delimiter=",", header=",".join(List_parameters))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_train.csv"), target_8D_train, delimiter=",", header=",".join(List_parameters))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_val.csv"), target_8D_val, delimiter=",", header=",".join(List_parameters))
    np.savetxt(os.path.join(args.output_dir_samples_8D, "target_test.csv"), target_8D_test, delimiter=",", header=",".join(List_parameters))

    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_train.csv"), original_3D_train, delimiter=",", header="Enu_true,ELep,CosLep")
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_val.csv"), original_3D_val, delimiter=",", header="Enu_true,ELep,CosLep")
    np.savetxt(os.path.join(args.output_dir_samples_3D, "original_test.csv"), original_3D_test, delimiter=",", header="Enu_true,ELep,CosLep")
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_train.csv"), target_3D_train, delimiter=",", header="Enu_true,ELep,CosLep")
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_val.csv"), target_3D_val, delimiter=",", header="Enu_true,ELep,CosLep")
    np.savetxt(os.path.join(args.output_dir_samples_3D, "target_test.csv"), target_3D_test, delimiter=",", header="Enu_true,ELep,CosLep")

    