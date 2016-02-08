import os
import numpy as np
import scipy.sparse
from scipy.special import digamma

from CPPLoader import LoadFuncFromCPPLib

curdir = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
sparseLocalStepManyDocs_cpp = LoadFuncFromCPPLib(
    os.path.join(curdir, 'libsparseManyDocs.so'),
    os.path.join(curdir, 'TopicModelLocalStepManyDocsCPPX.cpp'),
    'sparseLocalStepManyDocs_ActiveOnly')

def sparseLocalStepManyDocs(
        Data=None,
        alphaEbeta=None, alphaEbetaRem=None,
        ElogphiT=None,
        DocTopicCount=None,
        spResp_data_OUT=None,
        spResp_colids_OUT=None,
        nCoordAscentItersLP=10,
        convThrLP=0.001,
        nnzPerRowLP=2,
        restartLP=0,
        restartNumTrialsLP=0,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        reviseActiveFirstLP=-1,
        reviseActiveEveryLP=1,
        maxDiffVec=None,
        numIterVec=None,
        nRAcceptVec=None,
        nRTrialVec=None,
        **kwargs):
    ''' Perform local inference for topic model. Wrapper around C++ code.
    '''
    N = Data.nUniqueToken
    V, K = ElogphiT.shape
    assert K == alphaEbeta.size
    nnzPerRowLP = np.minimum(nnzPerRowLP, K)

    # Parse params for tracking convergence progress
    if maxDiffVec is None:
        maxDiffVec = np.zeros(Data.nDoc, dtype=np.float64)
        numIterVec = np.zeros(Data.nDoc, dtype=np.int32)
    if nRTrialVec is None:
        nRTrialVec = np.zeros(1, dtype=np.int32)
        nRAcceptVec = np.zeros(1, dtype=np.int32)
    assert maxDiffVec.dtype == np.float64
    assert numIterVec.dtype == np.int32
    
    # Use provided DocTopicCount array if its the right size
    # Otherwise, create a new one from scratch
    TopicCount_OUT = None
    if isinstance(DocTopicCount, np.ndarray):
        if DocTopicCount.shape == (Data.nDoc, K):
            TopicCount_OUT = DocTopicCount
    if TopicCount_OUT is None:
        TopicCount_OUT = np.zeros((Data.nDoc, K))
    assert TopicCount_OUT.shape == (Data.nDoc, K)
    if spResp_data_OUT is None:
        spResp_data_OUT = np.zeros(N * nnzPerRowLP)
        spResp_colids_OUT = np.zeros(N * nnzPerRowLP, dtype=np.int32)
    assert spResp_data_OUT.size == N * nnzPerRowLP
    assert spResp_colids_OUT.size == N * nnzPerRowLP

    if initDocTopicCountLP.startswith("setDocProbsToEGlobalProbs"):
        initProbsToEbeta = 1
    else:
        initProbsToEbeta = 0
    if reviseActiveFirstLP < 0:
        reviseActiveFirstLP = nCoordAscentItersLP + 10

    sparseLocalStepManyDocs_cpp(
        alphaEbeta, ElogphiT,
        Data.word_count, Data.word_id, Data.doc_range,
        nnzPerRowLP, N, K, Data.nDoc, Data.vocab_size,
        nCoordAscentItersLP, convThrLP,
        initProbsToEbeta,
        TopicCount_OUT,
        spResp_data_OUT,
        spResp_colids_OUT,
        numIterVec,
        maxDiffVec,
        restartNumTrialsLP * restartLP,
        nRAcceptVec,
        nRTrialVec,
        1)

    # Package results up into dict
    LP = dict()
    LP['DocTopicCount'] = TopicCount_OUT
    indptr = np.arange(
        0, (N+1) * nnzPerRowLP, nnzPerRowLP, dtype=np.int32)
    LP['spR'] = scipy.sparse.csr_matrix(
        (spResp_data_OUT, spResp_colids_OUT, indptr),
        shape=(N, K))
    # Fill in remainder of LP dict, with derived quantities
    from bnpy.allocmodel.topics.LocalStepManyDocs \
        import updateLPGivenDocTopicCount, writeLogMessageForManyDocs
    LP = updateLPGivenDocTopicCount(LP, LP['DocTopicCount'],
                                    alphaEbeta, alphaEbetaRem)
    LP['Info'] = dict()
    LP['Info']['iter'] = numIterVec
    LP['Info']['maxDiff'] = maxDiffVec

    if restartLP > 0:
        LP['Info']['nRestartsAccepted'] = nRAcceptVec[0]
        LP['Info']['nRestartsTried'] = nRTrialVec[0]
    writeLogMessageForManyDocs(Data, LP['Info'], **kwargs)
    return LP

if __name__ == '__main__':
    import bnpy
    model, Info = bnpy.run('BarsK10V900', 'HDPTopicModel', 'Mult', 'VB',
        nBatch=1, nDocTotal=1, nLap=2, initname='truelabels', 
        nCoordAscentItersLP=100, convThrLP=.01)
    LP0 = sparseLocalStepManyDocs(
        Data=Info['Data'],
        alphaEbeta=model.allocModel.alpha_E_beta(),
        alphaEbetaRem=model.allocModel.alpha_E_beta_rem(),
        ElogphiT=model.obsModel._E_logphiT('all'),
        nnzPerRowLP=4,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=10,
        convThrLP=-1,
        restartLP=0,
        )
    
    LP1 = sparseLocalStepManyDocs(
        Data=Info['Data'],
        alphaEbeta=model.allocModel.alpha_E_beta(),
        alphaEbetaRem=model.allocModel.alpha_E_beta_rem(),
        ElogphiT=model.obsModel._E_logphiT('all'),
        nnzPerRowLP=2,
        initDocTopicCountLP='setDocProbsToEGlobalProbs',
        nCoordAscentItersLP=100,
        convThrLP=-1,
        restartLP=1,
        )
    from IPython import embed; embed()
    