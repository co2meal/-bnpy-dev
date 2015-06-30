import numpy as np

def makeCandidateLPWithNewComps(
        Data_t, curLP_t, curModel, curSS_nott, **Plan):
    ''' 

    Returns
    -------
    propLP_t : local param dict, with K + Kx states
    '''
    # Execute proposal. Calls a function imported from BProposals.py
    creationProposalName = Plan['creationProposalName']
    propFuncName = 'expandLP_' + creationProposalName
    GlobalVars = globals()
    if propFuncName in GlobalVars:
        propFunc = GlobalVars[propFuncName]
        propLP_t = propFunc(Data_t, curLP_t, curModel, curSS_nott, **Plan)
    else:
        msg = "Unrecognized creationProposalName: %s" % (creationProposalName)
        raise NotImplementedError(msg)

    # Refine candidate local parameters
    propLP_t = refineCandidateViaLocalGlobalStepsAndDeletes(
            Data_t, propLP_t, propModel, curSS_nott,
            **Plan)

    return propLP_t



def refineCandidateViaLocalGlobalStepsAndDeletes(
        Data_t, propLP_t, propModel, curSS_nott,
        nRefineIters=3,
        verbose=0,
        **kwargs):
    ''' Improve proposed LP via conventional updates and delete moves.

    Args
    -------

    Returns
    -------
    '''
    curSS_nott = curSS_nott.copy()   
    propK = propLP_t['resp'].shape[-1]
    origK = curSS_nott.K

    # Initialize tempSS and tempModel
    # to be fully consistent with propLP_n
    propSS_t = propModel.get_global_suff_stats(Data_t, propLP_t)
    assert propSS_t.K == propK

    Kextra = propK - curSS_nott.K
    if Kextra > 0:
        curSS_nott.insertEmptyComps(Kextra)
    propSS = curSS_nott
    propSS += propSS_t
    propModel.update_global_params(propSS)

    # Refine via repeated local/global steps
    for step in xrange(nRefineIters):
        propLP_t = propModel.calc_local_params(Data_t)
        propSS -= propSS_t
        propSS_t = propModel.get_global_suff_stats(Data_t, propLP_t)
        propSS += propSS_t
        propModel.update_global_params(propSS)
    # Here, propSS and propModel are fully-consistent,
    # representing both propLP_t and curSS_nott
    assert propSS.K == propK
    assert propModel.obsModel.K == propK
    assert propModel.allocModel.K == propK

    # Perform "consistent" removal of empty components
    # This guarantees that tempModel and tempSS reflect
    # the whole dataset, including updated propLP_n/propSS_n
    # TODO

    assert propSS.getCountVec().sum() >= propSS_t.getCountVec().sum() - 1e-7
    return propLP_t
