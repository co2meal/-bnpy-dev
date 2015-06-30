import numpy as np

from BRefinery import makeCandidateLPWithNewComps

def runBirthMove(
        Data_b, curModel, curSS_notb, curLP_b, **Plan):
    ''' Execute birth move on provided dataset and model

    Args
    -----
    Data_b : dataset
    hmodel : HModel object, with K comps
    SS_notb : SuffStatBag object with K comps
    LP_b : dict of local params, with K comps

    Returns
    -------
    LP_b : dict of local params for current batch, with K + Kx comps
    '''
    curModel = curModel.copy()
    propModel = curModel.copy()

    # Create target dataset, and associated summaries
    Data_t, curLP_t = subsampleTargetDataset(Data_b, curLP_b, **Plan)
    curSS_b = curModel.get_global_summary_stats(
        Data_b, curLP_b, doPrecompEntropy=1)
    curSS_t = curModel.get_global_summary_stats(
        Data_t, curLP_t, doPrecompEntropy=1)
    curSS_nott = curSS_notb + curSS_b
    curSS_nott -= curSS_t

    # Evaluate current score
    curSS = curSS_b + curSS_notb
    curModel.update_global_params(curSS)
    curLscore = curModel.calc_evidence(SS=curSS)

    # Propose new local parameters for target set
    propLP_t = makeCandidateLPWithNewComps(
        Data_t, curLP_t, propModel, curSS_nott, **Plan)
    propSS_t = propModel.get_global_summary_stats(
        Data_t, propLP_t, doPrecompEntropy=1)
    propSS = propSS_t + curSS_nott
    propModel.update_global_params(propSS)
    propLscore = propModel.calc_evidence(SS=propSS)

    if propLscore > curLscore:
        # Accept
        propLP_b = fillLPWithTarget(curLP_b, propLP_t, **Plan)
        return propLP_b
    else:
        # Reject
        return curLP_b

