"""
This package contains methods to implement the external photoevaporation using the FRIED grid (Haworth et al., 2018) with the prescription of Sellek et al.(2022).
The setup_externalPhotoevaporation_FRIED(sim) function automatically implements all the required modifications to the Simulation object.
"""


from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import get_MassLoss_ResampleGrid
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import MassLoss_FRIED, TruncationRadius
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import PhotoEntrainment_Size
from dustpylib.photoevaporation.external.functions_externalPhotoevaporation import SigmaDot_ExtPhoto, SigmaDot_ExtPhoto_Dust


from dustpylib.photoevaporation.external.setup_externalPhotoevaporation import setup_externalPhotoevaporation_FRIED


__all__ = [
    "get_MassLoss_ResampleGrid",
    "MassLoss_FRIED",
    "TruncationRadius",
    "PhotoEntrainment_Size",
    "SigmaDot_ExtPhoto",
    "SigmaDot_ExtPhoto_Dust",
    "setup_externalPhotoevaporation_FRIED",
]
