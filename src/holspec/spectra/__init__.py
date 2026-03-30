"""
Spectra of Hodge Laplacians and spectral observables.

Provides HodgeLaplacianSpectra, the primary Stage 4 pipeline output, along
with the Spectrum container, eigendecomposition function, and validation
utilities.
"""

from .base import HodgeLaplacianSpectra
from .spectrum import Spectrum
from .eigensolvers import compute_eigendecomposition, validate_solver_params, VALID_SOLVER_NAMES
from .validation import validate_spectrum, ZERO_EIGENVALUE_TOL
