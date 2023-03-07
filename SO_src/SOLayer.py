import torch
import numpy as np
import libMG as mg
from TOLayer import TOLayer

#Fully based on TOLayer, add one function to reinitialize the SDF
class SOLayer(TOLayer):
    @staticmethod
    def redistance(rho):
        rho_new = SOLayer.sol.reinitialize(rho, 1e-3, 1000, False)
        return rho_new