// TopicModelLocalStepCPPX.cpp
//

// Define this symbol to enable runtime tests for allocations
#define EIGEN_RUNTIME_NO_MALLOC 

#include <math.h>
#include "Eigen/Dense"
#include <boost/math/special_functions/digamma.hpp>

using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ======================================================== visible externally

extern "C" {
    void sparseLocalStepSingleDoc(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        );   

    void sparseLocalStepSingleDocWithWordCounts(
        double* wordcounts_d_IN,
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        );   
}

// ======================================================== Custom Type Defs
// ========================================================
// Simple names for array types
typedef Matrix<double, Dynamic, Dynamic, RowMajor> Mat2D_d;
typedef Matrix<double, 1, Dynamic, RowMajor> Mat1D_d;
typedef Array<double, Dynamic, Dynamic, RowMajor> Arr2D_d;
typedef Array<double, 1, Dynamic, RowMajor> Arr1D_d;
typedef Array<int, 1, Dynamic, RowMajor> Arr1D_i;

// Simple names for array types with externally allocated memory
typedef Map<Mat2D_d> ExtMat2D_d;
typedef Map<Mat1D_d> ExtMat1D_d;
typedef Map<Arr2D_d> ExtArr2D_d;
typedef Map<Arr1D_d> ExtArr1D_d;
typedef Map<Arr1D_i> ExtArr1D_i;

void sparseLocalStepSingleDoc(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);
    // Temporary storage
    VectorXd ElogProb_d (K);
    VectorXd prevTopicCount_d (K);
    VectorXd logScores_n (K);
    VectorXd tempScores_n (K);
    //VectorXd resp_n (nnzPerRow);
    //VectorXi colids_n (nnzPerRow);
    prevTopicCount_d.fill(-1);

    for (int iter = 0; iter < nCoordAscentIterLP; iter++) {

        for (int k = 0; k < K; k++) {
            ElogProb_d(k) = boost::math::digamma(
                topicCount_d(k) + alphaEbeta(k));
        }
    
        topicCount_d.fill(0);
        // Step over each data atom
        for (int n = 0; n < N; n++) {
            int m = n * nnzPerRow;
            int argmax_n = 0;
            logScores_n(0) = ElogProb_d(0) + ElogLik_d(n,0);
            double maxScore_n = logScores_n(0);
            for (int k = 1; k < K; k++) {
                logScores_n(k) = ElogProb_d(k) + ElogLik_d(n,k);
                if (logScores_n(k) > maxScore_n) {
                    maxScore_n = logScores_n(k);
                    argmax_n = k;
                }
            }
            if (nnzPerRow == 1) {
                spResp_data(m) = 1.0;
                spResp_colids(m) = argmax_n;
                // Update topicCount_d
                topicCount_d(argmax_n) += 1.0;
            } else {
                // Find the top L entries in logScores_n
                // Copy current row over into a temp buffer
                std::copy(
                    logScores_n.data(),
                    logScores_n.data() + K,
                    tempScores_n.data());
                // Sort the data in the temp buffer (in place)
                std:nth_element(
                    tempScores_n.data(),
                    tempScores_n.data() + K - nnzPerRow,
                    tempScores_n.data() + K);
                // Walk thru this row and find the top "nnzPerRow" positions
                double pivotScore = tempScores_n(K - nnzPerRow);
                int nzk = 0;
                double sumResp_n = 0.0;
                for (int k = 0; k < K; k++) {
                    if (logScores_n(k) >= pivotScore) {
                        spResp_data(m + nzk) = \
                            exp(logScores_n(k) - maxScore_n);
                        spResp_colids(m + nzk) = k;
                        sumResp_n += spResp_data(m + nzk);
                        nzk += 1;
                    }
                }
                // Normalize for doc-topic counts
                for (int nzk = 0; nzk < nnzPerRow; nzk++) {
                    spResp_data(m + nzk) /= sumResp_n;
                    topicCount_d(spResp_colids(m + nzk)) += \
                        spResp_data(m + nzk);
                }
            }
        }
        // END ITERATION. Decide whether to quit early
        if (iter > 3 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            double maxDiff = 0.0;
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
                prevTopicCount_d(k) = topicCount_d(k); // copy over
            }
            //printf("iter %d  maxDiff %.3f\n", iter, maxDiff);
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
}


void sparseLocalStepSingleDocWithWordCounts(
        double* wordcounts_d_IN,
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr1D_d wc_d (wordcounts_d_IN, N);
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);
    // Temporary storage
    VectorXd ElogProb_d (K);
    VectorXd prevTopicCount_d (K);
    VectorXd logScores_n (K);
    VectorXd tempScores_n (K);
    prevTopicCount_d.fill(-1);

    for (int iter = 0; iter < nCoordAscentIterLP; iter++) {

        for (int k = 0; k < K; k++) {
            ElogProb_d(k) = boost::math::digamma(
                topicCount_d(k) + alphaEbeta(k));
        }
    
        topicCount_d.fill(0);
        // Step over each data atom
        for (int n = 0; n < N; n++) {
            int m = n * nnzPerRow;
            int argmax_n = 0;
            logScores_n(0) = ElogProb_d(0) + ElogLik_d(n,0);
            double maxScore_n = logScores_n(0);
            for (int k = 1; k < K; k++) {
                logScores_n(k) = ElogProb_d(k) + ElogLik_d(n,k);
                if (logScores_n(k) > maxScore_n) {
                    maxScore_n = logScores_n(k);
                    argmax_n = k;
                }
            }
            if (nnzPerRow == 1) {
                spResp_data(m) = 1.0;
                spResp_colids(m) = argmax_n;
                // Update topicCount_d
                topicCount_d(argmax_n) += wc_d(n);
            } else {
                // Find the top L entries in logScores_n
                // Copy current row over into a temp buffer
                std::copy(
                    logScores_n.data(),
                    logScores_n.data() + K,
                    tempScores_n.data());
                // Sort the data in the temp buffer (in place)
                std:nth_element(
                    tempScores_n.data(),
                    tempScores_n.data() + K - nnzPerRow,
                    tempScores_n.data() + K);
                // Walk thru this row and find the top "nnzPerRow" positions
                double pivotScore = tempScores_n(K - nnzPerRow);
                int nzk = 0;
                double sumResp_n = 0.0;
                for (int k = 0; k < K; k++) {
                    if (logScores_n(k) >= pivotScore) {
                        spResp_data(m + nzk) = \
                            exp(logScores_n(k) - maxScore_n);
                        spResp_colids(m + nzk) = k;
                        sumResp_n += spResp_data(m + nzk);
                        nzk += 1;
                    }
                }
                // Normalize for doc-topic counts
                for (int nzk = 0; nzk < nnzPerRow; nzk++) {
                    spResp_data(m + nzk) /= sumResp_n;
                    topicCount_d(spResp_colids(m + nzk)) += \
                        wc_d(n) * spResp_data(m + nzk);
                }
            }
        }
        // END ITERATION. Decide whether to quit early
        if (iter > 3 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            double maxDiff = 0.0;
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
                prevTopicCount_d(k) = topicCount_d(k); // copy over
            }
            //printf("iter %d  maxDiff %.3f\n", iter, maxDiff);
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
}
