"""This code aims to evaluate the performance of a trained model (with different modes included in the training) on a single mode.
It takes as input the model (in a pickle format), the original and target datasets (and the target swd distribution) to be tested on (which "Mode" column
contains only the specific mode to be tested), and the path to save the output plots and metrics. 
It returns 1D histograms with ratio of reweighting data over target, a Chi2 statistics, some SWD values and the associated p-value."""

#imports
import os
import pickle
import numpy as np
import argparse
import json
from Metrics_ndim import chi2_hist_axis, plot_histograms, chi2_dof, compute_swd, compute_p_value
from Train_predict import predict_XGB


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Evaluate the performance of a trained model on a different dataset.')
    argparser.add_argument('--model_path', type=str, required=False, help='Path to the trained model (pickle file).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_models/6D_mode1_11/6D/XGB_model_6D.pkl')
    argparser.add_argument('--original_test_nD', type=str, required=False, help='Path to the original 6D dataset (containing the desired mode) (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_samples/6D_mode1_11/6D/original_test.csv')
    argparser.add_argument('--original_test', type=str, required=False, help='Path to the original dataset with ALL PARAMETERS (containing the desired mode) (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_samples/6D_mode1_11/21D/original_test.csv')
    argparser.add_argument('--target_test_nD', type=str, required=False, help='Path to the target 6D dataset (containing the desired mode) (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_11/6D/target_test.csv')
    argparser.add_argument('--target_test', type=str, required=False, help='Path to the target dataset with ALL PARAMETERS (containing the desired mode) (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_samples/6D_5andmode_mode_11/21D/target_test.csv')
    argparser.add_argument('--swd_distribution_target_3D', type=str, required=False, help='Path to the SWD distribution of the target dataset for the 3D case (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_11/3D/swd_distribution_3D.npy')
    argparser.add_argument('--swd_distribution_target_8D', type=str, required=False, help='Path to the SWD distribution of the target dataset for the 8D case (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_11/8D/swd_distribution_8D.npy')
    argparser.add_argument('--swd_distribution_target_21D', type=str, required=False, help='Path to the SWD distribution of the target dataset for the 21D case (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_11/21D/swd_distribution_21D.npy')
    argparser.add_argument('--swd_distribution_target_nD', type=str, required=False, help='Path to the SWD distribution of the target dataset (csv).',
                            default='/home/hep/tlt26/RW_Snakemake/saved_swd_distribution/6D_5andmode_mode_11/6D/swd_distribution_6D.npy')
    argparser.add_argument('--binning_file', type=str, required=False, help="Path to the json file containing the binning information for each parameter.",
                           default="/home/hep/tlt26/RW_Snakemake/binnings.json")
    argparser.add_argument('--output_path', type=str, required=False, help='Path to save the output plots and metrics.',
                            default='/home/hep/tlt26/RW_Snakemake/Saved_evaluation/trained_1_11_eval_11/')
    args = argparser.parse_args()

    List_parameters = ["Enu_true", "ELep", "CosLep", "Q2", "q0", "q3", "W", "Eav", "y", "Mode",
                "cc", "hitnuc", "N_n", "K_n", "N_p", "K_p", "N_pi0", "K_pi0", "N_pip", "K_pip", "N_pim", "K_pim"]

    # Load data, model, binning and swd distribution
    original_test_nD = np.loadtxt(args.original_test_nD, delimiter=",")
    original_test = np.loadtxt(args.original_test, delimiter=",")
    target_test_nD = np.loadtxt(args.target_test_nD, delimiter=",")
    target_test = np.loadtxt(args.target_test, delimiter=",")
    with open(args.model_path, 'rb') as f:
        model = pickle.load(f)
    with open(args.binning_file, 'r') as f:
        binning_dict = json.load(f)

    swd_distribution_target_3D = np.load(args.swd_distribution_target_3D)
    swd_distribution_target_8D = np.load(args.swd_distribution_target_8D)
    swd_distribution_target_nD = np.load(args.swd_distribution_target_nD)
    swd_distribution_target_21D = np.load(args.swd_distribution_target_21D)

    # The model predicts weights
    weights = predict_XGB(original_test_nD, model)
    weights_dict = {"XGB": weights, "Unweighted": np.ones_like(weights)/len(weights)}
    # We plot using plot_histograms
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    plot_histograms(original_test, target_test, weights_dict,
                        dict_binning=binning_dict,
                        xlabels = List_parameters, 
                        output_file=os.path.join(args.output_path, "1Dhist.pdf"))
    
    # We compute the chi2 statistics and p-value for each parameter
    metrics_dict = {}
    chi2_dict = {}
    p_value_dict = {}
    for key in weights_dict.keys():
        chi2_dict[key] = {}
        for i, param in enumerate(List_parameters):
            x_min = binning_dict[param]["x_min"]
            x_max = binning_dict[param]["x_max"]
            n_bins = binning_dict[param]["n_bins"]
            chi2, dof = chi2_hist_axis(original_test, target_test, weights_dict[key], axis_number= i, x_min=x_min, x_max=x_max, n_bins=n_bins)
            chi2_dict[key][param] = chi2/dof
    metrics_dict["indiv_chi2_dof"] = chi2_dict
    chi2_3D = chi2_dof(original_test[:, :3], target_test[:, :3], weights_dict=weights_dict, binning_dict=binning_dict)
    metrics_dict["chi2_3D_dof"] = chi2_3D
    chi2_8D = chi2_dof(original_test[:, :8], target_test[:, :8], weights_dict=weights_dict, binning_dict=binning_dict)
    metrics_dict["chi2_8D_dof"] = chi2_8D
    chi2_21D = chi2_dof(original_test, target_test, weights_dict=weights_dict, binning_dict=binning_dict)
    metrics_dict["chi2_21D_dof"] = chi2_21D

    swd_3D = compute_swd(original_test[:, :3], target_test[:, :3], weights_dict)
    swd_8D = compute_swd(original_test[:, :8], target_test[:, :8], weights_dict)
    swd_21D = compute_swd(original_test, target_test, weights_dict)
    swd_nD = compute_swd(original_test_nD, target_test_nD, weights_dict)
    metrics_dict["swd_3D"] = swd_3D
    metrics_dict["swd_8D"] = swd_8D
    metrics_dict["swd_21D"] = swd_21D
    metrics_dict["swd_nD"] = swd_nD

    p_value_3D = compute_p_value(swd_3D, swd_distribution_target_3D)
    p_value_8D = compute_p_value(swd_8D, swd_distribution_target_8D)
    p_value_21D = compute_p_value(swd_21D, swd_distribution_target_21D)
    p_value_nD = compute_p_value(swd_nD, swd_distribution_target_nD)
    p_value_dict["p_value_3D"] = p_value_3D
    p_value_dict["p_value_8D"] = p_value_8D
    p_value_dict["p_value_21D"] = p_value_21D
    p_value_dict["p_value_nD"] = p_value_nD

    metrics_dict["p_values"] = p_value_dict

    # Save metrics in a json file
    
    with open(os.path.join(args.output_path, "metrics.json"), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

    

    

    

                            
                           
    