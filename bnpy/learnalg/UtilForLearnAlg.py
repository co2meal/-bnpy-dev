

def calcLocalParamsAndSummarize_singlethread(
        DataIterator, hmodel, learnAlg,
        SS_notb=None,
        batchID=0,
        LPkwargs=None, MergePrepInfo=None, 
        **kwargs):
    ''' Execute local step and summary step in main process.

    Returns
    -------
    SSbatch : bnpy.suffstats.SuffStatBag
        Aggregated suff stats representing specified batch.
    '''
    if not isinstance(LPkwargs, dict):
        LPkwargs = dict()
    if not isinstance(MergePrepInfo, dict):
        MergePrepInfo = dict()
    LPkwargs.update(MergePrepInfo)

    Data_b = DataIterator.getBatch(batchID=batchID)
    LP_b = hmodel.calc_local_params(Data_b, **LPkwargs)

    if learnAlg.hasMove('birth'):
        Plans = makeBirthPlans(Data_b, hmodel, SS_notb, LP_b, **birthKwargs)
        for Plan in Plans:
            Plan.update(birthKwargs)
            LP_b = runBirthMove(
                Data_b, hmodel, SS_notb, LP_b, **Plan)

    SS_b = hmodel.get_global_suff_stats(
        Data_b, LP_b, doPrecompEntropy=1, **MergePrepInfo)
    SS_b.setUIDs(SS_notb.uIDs.copy())
    return SS_b
