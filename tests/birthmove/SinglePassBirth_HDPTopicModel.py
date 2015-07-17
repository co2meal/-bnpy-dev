import numpy as np
import bnpy
import time

from bnpy.birthmove.SCreateFromScratch import createSplitStats
from bnpy.birthmove.SAssignToExisting import assignSplitStats
from bnpy.birthmove.SCreateFromScratch import DefaultLPkwargs
from matplotlib import pylab;

def main(nBatch=10, 
         nFixedInitLaps=0, Kinit=3, targetUID=0,
         creationProposalName='kmeans',
         **kwargs):
    LPkwargs = DefaultLPkwargs

    import BarsK10V900
    Data = BarsK10V900.get_data(nDocTotal=200, nWordsPerDoc=500)
    Data.alwaysTrackTruth = 1
    DataIterator = Data.to_iterator(nBatch=nBatch, nLap=10)

    hmodel = bnpy.HModel.CreateEntireModel(
        'moVB', 'HDPTopicModel', 'Mult',
        dict(alpha=0.5, gamma=10), dict(lam=0.1), Data)
    hmodel.init_global_params(
        DataIterator.getBatch(0), K=Kinit, initname='kmeansplusplus')

    # Do some fixed-truncation local/global steps
    SS = None
    SSmemory = dict()
    for lap in range(nFixedInitLaps):
        for batchID in xrange(nBatch):
            Dbatch = DataIterator.getBatch(batchID)

            LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
            SSbatch = hmodel.get_global_suff_stats(
                Dbatch, LPbatch, doPrecompEntropy=1, doTrackTruncationGrowth=1)

            if batchID in SSmemory:        
                SS -= SSmemory[batchID]
            SSmemory[batchID] = SSbatch
            if SS is None:
                SS = SSbatch.copy()
            else:
                SS += SSbatch
            hmodel.update_global_params(SS)

    for batchID in xrange(nBatch):
        print 'batch %d/%d' % (batchID+1, nBatch)
        Dbatch = DataIterator.getBatch(batchID)

        LPbatch = hmodel.calc_local_params(Dbatch, **LPkwargs)
        SSbatch = hmodel.get_global_suff_stats(
            Dbatch, LPbatch, doPrecompEntropy=1, doTrackTruncationGrowth=1)

        if batchID in SSmemory:        
            SS -= SSmemory[batchID]
        SSmemory[batchID] = SSbatch
        if SS is None:
            SS = SSbatch.copy()
        else:
            SS += SSbatch
        
        if batchID == 0:
            xSSbatch, propSSbatch = createSplitStats(
                Dbatch, hmodel, LPbatch, curSSwhole=SS,
                creationProposalName=creationProposalName,
                targetUID=targetUID,
                newUIDs=np.arange(100, 100+10),
                LPkwargs=LPkwargs,
                returnPropSS=1)

            xSS = xSSbatch.copy()
            propSS_agg = propSSbatch.copy()
        else:
            xSSbatch, propSSbatch = assignSplitStats(
                Dbatch, hmodel, LPbatch, SS, xSS,
                targetUID=targetUID,
                LPkwargs=LPkwargs,
                returnPropSS=1)
            xSS += xSSbatch
            propSS_agg += propSSbatch

        propSS_whole = propSS_agg.copy()
        for rembatchID in range(batchID+1, nBatch):
            if rembatchID in SSmemory:
                remSSbatch = SSmemory[rembatchID].copy()
                Kextra = propSS_whole.K - SSbatch.K
                if Kextra > 0:
                    remSSbatch.insertEmptyComps(Kextra)
                propSS_whole += remSSbatch

        hmodel.update_global_params(SS)

        if batchID < 10 or (batchID + 1) % 10 == 0:
            curLscore = hmodel.calc_evidence(SS=SS)

            propSS = SS.copy()
            propSS.transferMassFromExistingToExpansion(
                uid=targetUID, xSS=xSS)

            for field in ['sumLogPi', 'sumLogPiRemVec']:
                arr = getattr(propSS, field)
                arr_direct = getattr(propSS_whole, field)
                if not np.allclose(arr, arr_direct):
                    print '  Error detected in field: %s' % (field)
                    from IPython import embed; embed()
                print '  SS field %s verified' % (field)

            for field in ['gammalnTheta', 'slackTheta', 'slackThetaRem',
                          'gammalnSumTheta', 'gammalnThetaRem']:
                arr = getattr(propSS._ELBOTerms, field)
                arr_direct = getattr(propSS_whole._ELBOTerms, field)
                if not np.allclose(arr, arr_direct):
                    print '  Error detected in field: %s' % (field)
                    from IPython import embed; embed()
                print '  ELBO field %s verified' % (field)

            propModel = hmodel.copy()
            propModel.update_global_params(propSS)

            propLscore = propModel.calc_evidence(SS=propSS)

            assert np.allclose(SS.getCountVec().sum(),
                               propSS.getCountVec().sum())

            print ' curLscore %.3f' % (curLscore)
            print 'propLscore %.3f' % (propLscore)
            from IPython import embed; embed()
            highlightComps = np.hstack([targetUID, np.arange(Kinit, propSS.K)])
            if propLscore > curLscore:
                print 'ACCEPTED!'
                bnpy.viz.PlotComps.plotCompsFromHModel(propModel)
                pylab.show(block=False)
                time.sleep(2)
                raw_input(">>> Press key to continue")
            else:
                print 'REJECTED <<<<<<<<<< :('
                bnpy.viz.PlotComps.plotCompsFromHModel(
                    propModel, compsToHighlight=highlightComps)
                pylab.show(block=False)
                time.sleep(2)

if __name__ == '__main__':
    main()
