"""
The :mod:`init` module gathers initialization procedures for model parameters
"""
import FromScratchGauss, FromScratchMult
import FromScratchBernRel
import FromSaved, FromTruth
import FromScratchLocal

__all__ = ['FromScratchLocal','FromScratchGauss', 'FromSaved', 'FromTruth', 'FromScratchMult', 'FromScratchBernRel']
