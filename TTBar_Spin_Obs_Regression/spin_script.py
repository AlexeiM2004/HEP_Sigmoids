import awkward as ak
import vector 
import numpy as np
import uproot 

def boost(top,anti_top,leptonP,leptonM):

    """
    Performs the relevant boosts and scalar products to return the cos-variables
    """

    # Build ttbar system
    ttbar = top + anti_top 

    # Boost tops lab --> ttbar_CoM
    top_in_CoM      = top.boostCM_of(ttbar)
    antitop_in_CoM  = anti_top.boostCM_of(ttbar) 

    # Boost leptons lab --> ttbar_CoM --> parent_tops'_CoM
    leptonP_in_CoM     = leptonP.boostCM_of(ttbar)
    leptonM_in_CoM     = leptonM.boostCM_of(ttbar)
    leptonP_in_top     = leptonP_in_CoM.boostCM_of(top_in_CoM)
    leptonM_in_antitop = leptonM_in_CoM.boostCM_of(antitop_in_CoM)

    # Compute the unit 3-vector directions
    Kdirection        = top_in_CoM.to_beta3().unit()
    leptonP_direction = leptonP_in_top.to_beta3().unit()
    leptonM_direction = leptonM_in_antitop.to_beta3().unit()

    return Kdirection, leptonP_direction, leptonM_direction


def helicity_basis_observables(Kdirection, leptonP_direction, leptonM_direction):

    """
    The helicity basis is a basis used in the measurement of spin observables at
    the LHC
    """

    # Kinematics
    z     = vector.obj(x=0,y=0,z=1)
    cos_T = z.dot(Kdirection)
    sin_T = (1 - cos_T**2)**0.5
    mask = 1*(cos_T > 0) -1*(cos_T < 0)

    # Helicity basis observables
    cos_K_plus  = Kdirection.dot(leptonP_direction)
    cos_K_minus = -Kdirection.dot(leptonM_direction) 

    Ndirection = z.cross(Kdirection)/sin_T
    cos_N_plus  = mask*Ndirection.dot(leptonP_direction)
    cos_N_minus = -mask*Ndirection.dot(leptonM_direction) 

    Rdirection  =  1/sin_T * (z - cos_T*Kdirection)
    cos_R_plus  = mask*Rdirection.dot(leptonP_direction)
    cos_R_minus = -mask*Rdirection.dot(leptonM_direction) 

    cos_phi     = leptonP_direction.dot(leptonM_direction)

    helicity_observables = ak.zip({
    "cos_K_plus"  :  cos_K_plus,
    "cos_K_minus" :  cos_K_minus,
    "cos_N_plus"  :  cos_N_plus,
    "cos_N_minus" :  cos_N_minus,
    "cos_R_plus"  :  cos_R_plus,
    "cos_R_minus" :  cos_R_minus,
    "cos_phi"     :  cos_phi 
    }, depth_limit=1, with_name="Event")

    return helicity_observables


# def compute_spin_parameters(observable_array):

#     """
#     Computes the spin parameters (C_{i,j}, B_k^{+/-})
#     """

#     Ckk = -9*(np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy()).mean())
#     Cnn = -9*(np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy()).mean())
#     Crr = -9*(np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy()).mean())
#     Crk = -9*(np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy()).mean())
#     Ckr = -9*(np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy()).mean())
#     Cnr = -9*(np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy()).mean())
#     Crn = -9*(np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy()).mean())
#     Cnk = -9*(np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy()).mean())
#     Ckn = -9*(np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy()).mean())

#     CrkP = Crk + Ckr 
#     CrkM = Crk - Ckr 
#     CnrP = Cnr + Crn
#     CnrM = Cnr - Crn 
#     CnkP = Cnk + Ckn 
#     CknM = Cnk - Ckn

#     BkP = -3*observable_array["cos_K_plus"].to_numpy().mean()
#     BkM = -3*observable_array["cos_K_minus"].to_numpy().mean()
#     BnP = -3*observable_array["cos_N_plus"].to_numpy().mean()
#     BnM = -3*observable_array["cos_N_minus"].to_numpy().mean()
#     BrP = -3*observable_array["cos_R_plus"].to_numpy().mean()
#     BrM = -3*observable_array["cos_R_minus"].to_numpy().mean()

#     return {"Ckk"       : Ckk,
#             "Cnn"       : Cnn,
#             "Crr"       : Crr,
#             "CrkP"      : CrkP,
#             "CrkM"      : CrkM,
#             "CnrP"      : CnrP,
#             "CnrM"      : CnrM,
#             "CnkP"      : CnkP,
#             "CknM"      : CknM,
#             "BkP"       : BkP,
#             "BkM"       : BkM,
#             "BnP"       : BnP,
#             "BnM"       : BnM,
#             "BrP"       : BrP,
#             "BrM"       : BrM}


# def histograms(observable_array, Nbins:int=10 ):

#     """
#     Use the cosvariable arrays to build the relevant angular observable
#     histograms
#     Args:
#     - Observable_array: awkward-array of the cos-variables
#     - Nbins: integer defining the binning (optional)
#     """

#     import boost_histogram as bh

#     DH = {  "Ckk"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "Cnn"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "Crr"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CrkP"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CrkM"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CnrP"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CnrM"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CnkP"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "CknM"      : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BkP"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BkM"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BnP"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BnM"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BrP"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "BrM"       : bh.Histogram(bh.axis.Regular(Nbins, -1, +1)),
#             "cos_phi"   : bh.Histogram(bh.axis.Regular(Nbins, -1, +1))
#     }

#     # Diagonal terms
#     DH["Ckk"].fill(np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy()))
#     DH["Cnn"].fill(np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy()))
#     DH["Crr"].fill(np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy()))

#     # Cross-terms
#     Crk = np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy())
#     Ckr = np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy())
#     Cnr = np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_R_minus"].to_numpy())
#     Crn = np.multiply(observable_array["cos_R_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy())
#     Cnk = np.multiply(observable_array["cos_N_plus"].to_numpy() , observable_array["cos_K_minus"].to_numpy())
#     Ckn = np.multiply(observable_array["cos_K_plus"].to_numpy() , observable_array["cos_N_minus"].to_numpy())

#     DH["CrkP"].fill(np.add(Crk,Ckr))
#     DH["CrkM"].fill(np.add(Crk,-Ckr))
#     DH["CnrP"].fill(np.add(Cnr,Crn))
#     DH["CnrM"].fill(np.add(Cnr,-Crn))
#     DH["CnkP"].fill(np.add(Cnk,Ckn))
#     DH["CknM"].fill(np.add(Cnk,-Ckn))

#     # Polarisations
#     DH["BkP"].fill(observable_array["cos_K_plus"].to_numpy())
#     DH["BkM"].fill(observable_array["cos_K_minus"].to_numpy())
#     DH["BnP"].fill(observable_array["cos_N_plus"].to_numpy())
#     DH["BnM"].fill(observable_array["cos_N_minus"].to_numpy())
#     DH["BrP"].fill(observable_array["cos_R_plus"].to_numpy())
#     DH["BrM"].fill(observable_array["cos_R_minus"].to_numpy())

#     DH["cos_phi"].fill(observable_array["cos_phi"].to_numpy())

#     return DH


def main():
    f = uproot.open("Downloads/ttbar_2L_mc20eTest50_240426A_410472_mc20e_fullsim.root")
    

    Parton_Top = vector.zip({
        "pt": f["reco"]["parton_top_pt"].array(),
        "eta": f["reco"]["parton_top_eta"].array(),
        "phi": f["reco"]["parton_top_phi"].array(),
        "mass": f["reco"]["parton_top_m"].array()
    })

    Parton_AntiTop = vector.zip({
        "pt": f["reco"]["parton_antitop_pt"].array(),
        "eta": f["reco"]["parton_antitop_eta"].array(),
        "phi": f["reco"]["parton_antitop_phi"].array(),
        "mass": f["reco"]["parton_antitop_m"].array()
    })

    Parton_leptonPlus = vector.zip({
        "pt": f["reco"]["parton_lepton_plus_pt"].array(),
        "eta": f["reco"]["parton_lepton_plus_eta"].array(),
        "phi": f["reco"]["parton_lepton_plus_phi"].array(),
        "mass": f["reco"]["parton_lepton_plus_m"].array() 
    })

    Parton_leptonMinus = vector.zip({
        "pt": f["reco"]["parton_lepton_minus_pt"].array(),
        "eta": f["reco"]["parton_lepton_minus_eta"].array(),
        "phi": f["reco"]["parton_lepton_minus_phi"].array(),
        "mass": f["reco"]["parton_lepton_minus_m"].array()  
})
    
    dic_of_spin_observables = helicity_basis_observables(*boost(Parton_Top,
                                                                Parton_AntiTop,
                                                                Parton_leptonPlus,
                                                                Parton_leptonMinus))