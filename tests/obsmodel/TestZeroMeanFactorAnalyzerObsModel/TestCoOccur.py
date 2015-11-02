import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat, savemat
from scipy.linalg import cholesky, solve_triangular
from math import floor
import bnpy
from bnpy.data import XData

nBatch = 50
modelName = 'C50/10/'

dataPath = '/Users/Geng/Documents/Brown/research/patch/HDP_patches/BerkSeg500/Patches_Size8x8_Stride4_recover/'
modelPath = '/Users/Geng/Documents/Brown/research/patch/FAPY/Half/' + modelName
savePath = '/Users/Geng/Documents/Brown/research/patch/coOccurTest/' + modelName + 'recover/'


def build_img_from_lst(lst, H, W, patchH, patchW, pace):
    '''
    Reconstruct an image-like 2D array based on a list of f(patch)
    '''
    assert patchH % pace == 0 and patchW % pace == 0
    assert H % patchH == 0 and W % patchW == 0
    nRow = H/patchH
    nCol = W/patchW
    RIm = np.zeros((H/pace, W/pace))
    for patchID in xrange(nRow * nCol):
        # c is integer in 0, 1, ... nCol - 1
        c = floor(patchID / nRow)
        # r is integer in 0, 1, ... nRow - 1
        r = patchID % nRow
        # Reconstruct
        r = r * patchH / pace
        c = c * patchW / pace
        for i in xrange(patchH/pace):
            for j in xrange(patchW/pace):
                RIm[r+i, c+j] = lst[patchID + (i*patchW/pace+j) * nRow * nCol]
    return RIm

def build_co_occur_mat(img, dir, K):
    result = np.zeros((K,K))
    H, W = img.shape
    if dir == 0:
        for r in xrange(H):
            for c in xrange(W-1):
                result[img[r,c], img[r,c+1]] += 1
    elif dir == 45:
        for r in xrange(1, H):
            for c in xrange(W-1):
                result[img[r,c], img[r-1,c+1]] += 1
    elif dir == 90:
        for r in xrange(1, H):
            for c in xrange(W):
                result[img[r,c], img[r-1,c]] += 1
    elif dir == 135:
        for r in xrange(1, H):
            for c in xrange(1, W):
                result[img[r,c], img[r-1,c-1]] += 1
    else:
        raise NotImplementedError
    return result


def main(direction, reorder=True, noiseStd=None):
    hmodel = bnpy.load_model(modelPath)
    K = hmodel.obsModel.K
    w = hmodel.allocModel.get_active_comp_probs()
    if reorder:
        kMap = dict(zip(np.argsort(-1 * w), np.arange(K)))
    if noiseStd is not None:
        noiseStd /= 255
        logW = np.log(w)
        D = hmodel.obsModel.D
        cholCovs = np.zeros((K, D, D))
        logdetCovs = np.zeros(K)
        for k in xrange(K):
            Cov = hmodel.obsModel.getGaussCov4Comp(k=k) + noiseStd**2 * np.eye(D)
            cholCovs[k] = cholesky(Cov, lower=1)
            logdetCovs[k] = 2 * np.sum(np.log(np.diag(cholCovs[k])))
    COMat = np.zeros((K,K))
    for b in xrange(nBatch):
        print 'processing batch %3d' % b
        rawData = loadmat(dataPath + 'Batch%03d.mat'%(b+1))
        if 'stride' not in rawData:
            stride = 4
        if 'patchH' not in rawData:
            patchH = 8
            patchW = 8
        docRange = rawData['doc_range'][0]
        for d in xrange(int(rawData['nImgPerBatch'])):
            idx_start = docRange[d]
            idx_end = docRange[d+1]
            Data = XData(rawData['X'][idx_start:idx_end])
            if noiseStd is None:
                LP = hmodel.calc_local_params(Data)
                kMaxX = np.argmax(LP['resp'], axis=1)
            else:
                resp = np.zeros((Data.nObs, K))
                for k in xrange(K):
                    resp[:,k] = logW[k] \
                                - 0.5 * logdetCovs[k] \
                                - 0.5 * _mahalDist(Data.X, cholCovs[k])
                kMaxX = np.argmax(resp, axis=1)
            if reorder:
                kMaxX = map(lambda x: kMap[x], kMaxX)
            H = rawData['BatchEffImgSizes'][d, 0]
            W = rawData['BatchEffImgSizes'][d, 1]
            kMaxImg = build_img_from_lst(kMaxX, H, W, patchH, patchW, stride)
            COMat += build_co_occur_mat(kMaxImg, direction, K)
    result = COMat / np.sum(COMat, axis=1)[:,np.newaxis]
    if noiseStd is not None:
        suffix = '_std%d' % (noiseStd*255)
    savemat(savePath + 'COMat_%d' % direction + suffix, dict(COMat=result))

def _mahalDist(X, cholCov):
    Q = solve_triangular(cholCov, X.T, lower=True)  # zero mean assumed here!
    Q *= Q
    return np.sum(Q, axis=0)

if __name__ == '__main__':
    direction = [0, 45, 90, 135]
    for i in direction:
        print 'Direction: %d' % i
        main(i, noiseStd=25.0)

    # k = 10
    # hmodel = bnpy.load_model(modelPath)
    # PRNG = np.random.RandomState(0)
    # sample = hmodel.obsModel.sampleFromComp(k=k)
    # plt.imshow(np.reshape(sample,(8,8),order='F'),extent=[0, 1, 0, 1])
    # cbar=plt.colorbar()
    # cbar.ax.tick_params(labelsize=30)
    # plt.savefig('/Users/Geng/Documents/Brown/research/patch/NewSemester/figures/%d.pdf'%(k+1), bbox_inches='tight')
    # plt.show()