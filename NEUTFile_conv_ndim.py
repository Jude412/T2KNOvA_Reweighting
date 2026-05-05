"""The objective of this script is to create a function that takes as input a Tree ROOT file from NEUT/T2KND and to 
return an array containing the desired parameters of interest. They are to be given as a list containing strings. One may also
include the weights of the events in the "weights" argument."""

#imports    
import numpy as np
import uproot
import awkward as ak

def convert_NEUT_input_file_ndim(input_file, mode, neutrino_PDG, list_parameters = ["Enu_true","ELep", "CosLep"], weights = None):
    """This function takes as input a ROOT file from NEUT/T2KND and returns an array containing the desired parameters of interest."""
    # Open the ROOT file
    file = uproot.open(input_file)
    branches = ["Mode", "PDGnu"] + list_parameters
    tree = file["FlatTree_VARS"].arrays(branches, library="ak")
    # we want to extract the needed mode and neutrino type
    mask = (tree["Mode"] == mode) & (tree["PDGnu"] == neutrino_PDG)
    mode_tree = tree[mask]
    # we now compute the parameters of interest
    # watch out, paramters such as cos(theta) may need to be converted later on
    param_values = []
    for param in list_parameters:
        param_values.append(ak.to_numpy(mode_tree[param]))
    data = np.array(np.column_stack(param_values))
    ## Check for NaN
    for i in range(data.shape[1]):
        if np.isnan(data[:, i]).any():
            print(f"Warning: NaN values found in column {i} of the data array.")
            data = data[~np.isnan(data[:, i])].copy()
        else:
            pass
    if weights is not None:
        weights = ak.to_numpy(mode_tree[weights])
        return data, weights
    else:
        return data