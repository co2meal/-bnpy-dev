import numpy as np

from BProposals import *

def makeCandidateLPWithNewComps(
        Data_t, curLP_t, propModel, curSS_nott, **Plan):
    ''' 

    Returns
    -------
    propLP_t : local param dict, with K + Kx states
    xcurSS_nott : SuffStatBag, with K + Kx states
        exactly equal to curSS_nott, with empties for states k >= K
    '''
    # Execute proposal. Calls a function imported from BProposals.py
    creationProposalName = Plan['bcreationProposalName']
    propFuncName = 'expandLP_' + creationProposalName
    GlobalVars = globals()
    if propFuncName in GlobalVars:
        propFunc = GlobalVars[propFuncName]
        propLP_t, xcurSS_nott = propFunc(
            Data_t, curLP_t, propModel, curSS_nott, **Plan)
    else:
        msg = "Unrecognized creationProposalName: %s" % (creationProposalName)
        raise NotImplementedError(msg)

    # Refine candidate local parameters
    propLP_t, xcurSS_nott = refineCandidateViaLocalGlobalStepsAndDeletes(
            Data_t, propLP_t, propModel, curSS_nott, xcurSS_nott,
            **Plan)

    return propLP_t, xcurSS_nott



def refineCandidateViaLocalGlobalStepsAndDeletes(
        Data_t, propLP_t, propModel, curSS_nott, xcurSS_nott,
        bRefineIters=3,
        verbose=0,
        **kwargs):
    ''' Improve proposed LP via conventional updates and delete moves.

    Args
    -------

    Returns
    -------
    '''
    origK = curSS_nott.K
    propK = propLP_t['resp'].shape[-1]

    # Initialize propSS and propModel
    # to be fully consistent with propLP_n
    propSS_t = propModel.get_global_suff_stats(Data_t, propLP_t)
    assert propSS_t.K == propK

    nottSize = xcurSS_nott.getCountVec().sum() 
    tSize = propSS_t.getCountVec().sum()

    propSS = xcurSS_nott
    assert propSS.K == propK

    propSS += propSS_t
    print ' '.join(['%.1f' % x for x in propSS.N[origK:]])
    propModel.update_global_params(propSS)
    propSS -= propSS_t

    # Refine via repeated local/global steps
    for step in xrange(bRefineIters):
        propLP_t = propModel.calc_local_params(Data_t)
        propSS_t = propModel.get_global_suff_stats(Data_t, propLP_t)

        propSS += propSS_t
        wholeSize = propSS.getCountVec().sum()
        assert np.allclose(wholeSize, tSize + nottSize, atol=1e-6, rtol=0)
        print ' '.join(['%.1f' % x for x in propSS.N[origK:]])
        propModel.update_global_params(propSS)
        propSS -= propSS_t


    # Here, propSS and propModel are fully-consistent,
    # representing both propLP_t and curSS_nott
    assert propSS.K == propK
    assert propModel.obsModel.K == propK
    assert propModel.allocModel.K == propK

    # Remove empty/redundant components
    # TODO
    return propLP_t, xcurSS_nott
