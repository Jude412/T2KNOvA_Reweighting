"""The objective of this script is to create a function that takes as input a FlatTree ROOT file from Nuisance (that used GENIE as generator) and to 
return an array containing the parameters of interest. They are to be given as a list containing strings. One may also include the weights of 
the events as parameters of interest in "weights"."""

#imports
import numpy as np
import uproot
import awkward as ak

def convert_GENIE_input_file_ndim(input_file, mode, neutrino_PDG, list_parameters = ["Enu_true", "ELep", "CosLep"], weights = None):
    """This function takes as input a ROOT file from Nuisance (that used GENIE as generator) and returns an array containing the desired parameters of interest."""
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
    
def convert_GENIE_input_file_alldim(input_file, modes = None):
    """This function takes as input a ROOT FlatTree from Nuisance and returns an array containing all the parameters of interest.
    They are : E_nu, E_lep, cos(theta_lep), Q2, q0, q3, W, Eav, y, neutrino PDG, interaction, mode, charged current,
    hit nucleus atomic number, hit nucleon PDG, multiplicity of final state proton, neutron, pions and the sum of their kinetic energies. """
    # Open the ROOT file
    file = uproot.open(input_file)
    branches = ["Enu_true", "ELep", "CosLep", "Eav", "Q2", "q0", "q3", "W", "y",
                 "PDGnu", "Mode", "cc", "nfsp", "px", "py", "pz", "E", "pdg", "pdg_vert"]
    List_final_params = ["Enu_true", "ELep", "CosLep", "Q2", "q0", "q3", "W", "Eav", "y",
                 "PDGnu", "Mode", "cc", "hitnuc", "A", "N_n", "K_n", "N_p", "K_p", "N_pi0", "K_pi0", "N_pip", "K_pip", "N_pim", "K_pim"]
    tree = file["FlatTree_VARS"].arrays(branches, library="ak")
    
    # we now compute the parameters of interest
    # A (this may change, since there might be different nuclei in the same file later on)
    coherent_mask = tree["Mode"] == 16
    A_value_string = np.unique(tree[coherent_mask]["pdg_vert"][:, 1])[0] # format : 10LZZZAAAI
    A_value = int(str(A_value_string)[3:6])
    tree["A"] = A_value
    # hitnucleon
    coherent_like_mask = ((tree["Mode"] == 16) | (tree["Mode"] == 36) | (tree["Mode"] == 2) | (tree["Mode"] == 32))
    tree["hitnuc"] = ak.where(
        coherent_like_mask,
        -999,
        ak.where(tree["pdg_vert"][:, 1] == 1000080160,
                 tree["pdg_vert"][:, 2],
             tree["pdg_vert"][:, 1])
                )
    #cut the tree to the desired modes if specified
    if modes is not None:
        mask = False
        for m in modes:
            mask = mask | (tree["Mode"] == m)

        tree = tree[mask]

    # multiplicity and sum of kinetic energies of final state protons, neutrons and pions
    pdg = tree["pdg"]
    E   = tree["E"]
    px = tree["px"]
    py = tree["py"]
    pz = tree["pz"]

    #Multiplicity
    tree["N_n"]   = ak.sum(pdg == 2112, axis=1)
    tree["N_p"]   = ak.sum(pdg == 2212, axis=1)
    tree["N_pi0"] = ak.sum(pdg == 111, axis=1)
    tree["N_pim"] = ak.sum(pdg == -211, axis=1)
    tree["N_pip"] = ak.sum(pdg == 211, axis=1)

    #Sum of kinetic energy
    tree["E_N"]   = E * (pdg == 2112)
    tree["E_P"]   = E * (pdg == 2212)
    tree["E_pi0"] = E * (pdg == 111)
    tree["E_pim"] = E * (pdg == -211)
    tree["E_pip"] = E * (pdg == 211)

    tree["P2_N"] = px**2 * (pdg == 2112) + py**2 * (pdg == 2112) + pz**2 * (pdg == 2112)
    tree["P2_P"] = px**2 * (pdg == 2212) + py**2 * (pdg == 2212) + pz**2 * (pdg == 2212)
    tree["P2_pi0"] = px**2 * (pdg == 111) + py**2 * (pdg == 111) + pz**2 * (pdg == 111)
    tree["P2_pim"] = px**2 * (pdg == -211) + py**2 * (pdg == -211) + pz**2 * (pdg == -211)
    tree["P2_pip"] = px**2 * (pdg == 211) + py**2 * (pdg == 211) + pz**2 * (pdg == 211)

    tree["K_n"]   = np.sum(tree["E_N"] - np.sqrt(tree["E_N"]**2 - tree["P2_N"]), axis=1)
    tree["K_p"]   = np.sum(tree["E_P"] - np.sqrt(tree["E_P"]**2 - tree["P2_P"]), axis=1)
    tree["K_pi0"] = np.sum(tree["E_pi0"] - np.sqrt(tree["E_pi0"]**2 - tree["P2_pi0"]), axis=1) 
    tree["K_pim"] = np.sum(tree["E_pim"] - np.sqrt(tree["E_pim"]**2 - tree["P2_pim"]), axis=1)
    tree["K_pip"] = np.sum(tree["E_pip"] - np.sqrt(tree["E_pip"]**2 - tree["P2_pip"]), axis=1)

    # we now create the final array containing the parameters of interest
    param_values = []
    for param in List_final_params:
        param_values.append(ak.to_numpy(tree[param]))
    data = np.array(np.column_stack(param_values))
    ## Check for NaN
    for i in range(data.shape[1]):
        if np.isnan(data[:, i]).any():
            print(f"Warning: NaN values found in column {i} of the data array.")
            data = data[~np.isnan(data[:, i])].copy()
        else:
            pass
    return data








