// TopicModelLocalStepCPPX.cpp
// Define this symbol to enable runtime tests for allocations
#define EIGEN_RUNTIME_NO_MALLOC 

#include <math.h>
#include "Eigen/Dense"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

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
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT
        );

    void sparseLocalStepSingleDoc_ActiveOnly(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        //double* digamma_alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int doTrackELBO,
        double* elboVec_OUT
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
        int initProbsToEbeta,
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


double calcELBOForSingleDoc_V2(
    Arr1D_d alphaEbeta,
    Arr1D_d topicCount_d,
    Arr1D_d spResp_data,
    Arr1D_i spResp_colids,
    Arr2D_d ElogLik_d,
    int K,
    int N,
    int nnzPerRow
    )
{
    double ELBO = 0.0;
    // Lalloc
    for (int k = 0; k < K; k++) {
        ELBO += boost::math::lgamma(topicCount_d(k) + alphaEbeta(k));
    }
    
    // Ldata and Lentropy
    for (int n = 0; n < N; n++) {
        for (int nzk = n * nnzPerRow; nzk < (n+1) * nnzPerRow; nzk++) {
            int k = spResp_colids(nzk);
            ELBO += spResp_data(nzk) * ElogLik_d(n,k);
            if (spResp_data(nzk) > 1e-9) {
                ELBO -= spResp_data(nzk) * log(spResp_data(nzk));
            }
        }
    }
    return ELBO;
}

double calcELBOForSingleDoc_V1(
    Arr1D_d alphaEbeta,
    Arr1D_d topicCount_d,
    Arr1D_d ElogProb_d,
    Arr1D_i activeTopics_d,
    double totalLogSumResp,
    double sum_gammalnalphaEbeta,
    int Kactive
    )
{
    // Compute Lalloc = \sum_k gammaln(\theta_dk)
    double ELBO = sum_gammalnalphaEbeta;
    for (int ka = 0; ka < Kactive; ka++) {
        int k = activeTopics_d(ka);
        ELBO += (
            boost::math::lgamma(topicCount_d(k) + alphaEbeta(k))
            - boost::math::lgamma(alphaEbeta(k))
            - topicCount_d(k) * ElogProb_d(k)
            );
    }
    ELBO += totalLogSumResp;
    return ELBO;
}


double stepForward_ActiveOnly(
    Arr1D_d alphaEbeta,
    Arr1D_d topicCount_d,
    Arr1D_d ElogProb_d,
    Arr2D_d ElogLik_d,
    Arr1D_d logScores_n,
    Arr1D_d tempScores_n,
    Arr1D_i activeTopics_d,
    Arr1D_d spResp_data,
    Arr1D_i spResp_colids,
    const int N,
    const int K, 
    const int Kactive,
    const int nnzPerRow,
    const int iter,
    const int initProbsToEbeta,
    const int doTrackELBO
    )
{
    double totalLogSumResp = 0.0;

    // UPDATE ElogProb_d using input doc-topic counts
    if (iter == 0 and initProbsToEbeta == 1) {
        assert(Kactive == K);
        for (int k = 0; k < Kactive; k++) {
            ElogProb_d(k) = log(alphaEbeta(k));
        }
    } else {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            ElogProb_d(k) = boost::math::digamma(
                topicCount_d(k) + alphaEbeta(k));
        }
    }

    // Reset topicCounts to all zeros
    topicCount_d.fill(0);

    // UPDATE sparse assignments
    for (int n = 0; n < N; n++) {
        int m = n * nnzPerRow;
        int argmax_n = 0;
        double maxScore_n;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            logScores_n(ka) = ElogProb_d(k) + ElogLik_d(n,k);
            if (ka == 0 || logScores_n(ka) > maxScore_n) {
                maxScore_n = logScores_n(ka);
                if (nnzPerRow == 1) {
                    argmax_n = k;
                }
            }
        }

        if (nnzPerRow == 1) {
            spResp_data(m) = 1.0;
            spResp_colids(m) = argmax_n;
            topicCount_d(argmax_n) += 1.0;
            if (doTrackELBO) {
                totalLogSumResp += maxScore_n;
            }
        } else {
            // Find the top L entries in logScores_n
            // Copy current row over into a temp buffer
            std::copy(
                logScores_n.data(),
                logScores_n.data() + Kactive,
                tempScores_n.data());
            // Sort the data in the temp buffer (in place)
            std:nth_element(
                tempScores_n.data(),
                tempScores_n.data() + Kactive - nnzPerRow,
                tempScores_n.data() + Kactive);
            // Walk thru this row and find the top "nnzPerRow" positions
            double pivotScore = tempScores_n(Kactive - nnzPerRow);

            int nzk = m;
            double sumResp_n = 0.0;
            for (int ka = 0; ka < Kactive; ka++) {
                if (logScores_n(ka) >= pivotScore) {
                    spResp_data(nzk) = \
                        exp(logScores_n(ka) - maxScore_n);
                    spResp_colids(nzk) = activeTopics_d(ka);
                    sumResp_n += spResp_data(nzk);
                    nzk += 1;                        
                }
            }

            for (int nzk = m; nzk < m + nnzPerRow; nzk++) {
                spResp_data(nzk) /= sumResp_n;
                topicCount_d(spResp_colids(nzk)) += \
                    spResp_data(nzk);
            }

            if (doTrackELBO) {
                totalLogSumResp += maxScore_n + log(sumResp_n);
            }
        }
    } // end for loop over tokens n
    return totalLogSumResp;
}

void sparseLocalStepSingleDoc(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);

    ExtArr1D_i numIterVec (numIterVec_OUT, D);
    ExtArr1D_d maxDiffVec (maxDiffVec_OUT, D);

    // Temporary storage
    Arr1D_d ElogProb_d (K);
    Arr1D_d prevTopicCount_d (K);
    Arr1D_d logScores_n (K);
    Arr1D_d tempScores_n (K);

    prevTopicCount_d.fill(-1);
    double maxDiff = N;
    int iter = 0;

    for (iter = 0; iter < nCoordAscentIterLP + initProbsToEbeta; iter++) {

        if (iter == 0 and initProbsToEbeta == 1) {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = log(alphaEbeta(k));
            }
        } else {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = boost::math::digamma(
                    topicCount_d(k) + alphaEbeta(k));
            }
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
        if (iter > 0 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            maxDiff = 0.0;
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
                prevTopicCount_d(k) = topicCount_d(k); // copy over
            }
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
    maxDiffVec(d) = maxDiff;
    numIterVec(d) = iter + 1;
}



void sparseLocalStepSingleDoc_ActiveOnly(
        double* ElogLik_d_IN,
        double* alphaEbeta_IN,
        //double* digamma_alphaEbeta_IN,
        int nnzPerRow,
        int N,
        int K,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_d_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int d,
        int D,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int doTrackELBO,
        double* elboVec_OUT
        )
{
    // Unpack inputs, treated as fixed constants
    ExtArr2D_d ElogLik_d (ElogLik_d_IN, N, K);
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    //ExtArr1D_d digamma_alphaEbeta (digamma_alphaEbeta_IN, K);

    // Unpack outputs
    ExtArr1D_d spResp_data (spResp_data_OUT, N * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, N * nnzPerRow);
    ExtArr1D_d topicCount_d (topicCount_d_OUT, K);

    ExtArr1D_i numIterVec (numIterVec_OUT, D);
    ExtArr1D_d maxDiffVec (maxDiffVec_OUT, D);
    ExtArr1D_d elboVec (elboVec_OUT, nCoordAscentIterLP + initProbsToEbeta);

    // Temporary storage
    Arr1D_d ElogProb_d (K);
    Arr1D_d prevTopicCount_d (K);
    Arr1D_d logScores_n (K);
    Arr1D_d tempScores_n (K);

    Arr1D_i activeTopics_d (K);
    Arr1D_i spareActiveTopics_d (nnzPerRow);

    prevTopicCount_d.fill(N);
    double maxDiff = N;
    int iter = 0;
    int Kactive = K;
    double ACTIVE_THR = 1e-9;

    double totalLogSumResp = 0.0;
    double sum_gammalnalphaEbeta = 0.0;
    if (doTrackELBO) {
        for (int k = 0; k < K; k++) {
            sum_gammalnalphaEbeta += boost::math::lgamma(alphaEbeta(k));
        }
    }

    for (iter = 0; iter < nCoordAscentIterLP + initProbsToEbeta; iter++) {
        totalLogSumResp = 0.0;
        if (iter > 0) {
            int newKactive = 0;
            int ia = 0; // spare inactive topics
            for (int a = 0; a < Kactive; a++) {
                int k = activeTopics_d(a);
                if (topicCount_d(k) > ACTIVE_THR) {
                    activeTopics_d(newKactive) = k;
                    prevTopicCount_d(k) = topicCount_d(k);
                    newKactive += 1;
                } else if (newKactive < nnzPerRow - ia) {
                    spareActiveTopics_d(ia) = k;
                    ia += 1;
                }
            }
            Kactive = newKactive;
            while (Kactive < nnzPerRow) {
                int k = spareActiveTopics_d(Kactive - newKactive);
                activeTopics_d(Kactive) = k;
                prevTopicCount_d(k) = ACTIVE_THR;
                Kactive++;
            }
            //std::copy(
            //    topicCount_d.data(),
            //    topicCount_d.data() + K,
            //    prevTopicCount_d.data());
        } else {
            for (int k = 0; k < K; k++) {
                activeTopics_d(k) = k;
            }
        }
        //printf("iter %3d: %3d active topics\n ", iter, Kactive);
        //for (int ka = 0; ka < Kactive; ka++) {
        //    int k = activeTopics_d(ka);
        //    printf("%3d:%5.1f ", k, topicCount_d(k));
        //}
        //printf("\n");
        assert(Kactive >= nnzPerRow);
        assert(Kactive <= K);

        if (iter == 0 and initProbsToEbeta == 1) {
            assert(Kactive == K);
            for (int k = 0; k < Kactive; k++) {
                ElogProb_d(k) = log(alphaEbeta(k));
            }
        } else {
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                ElogProb_d(k) = boost::math::digamma(
                    topicCount_d(k) + alphaEbeta(k));
            }
            /*
            Kactive = 0;
            for (int k = 0; k < K; k++) {
                if (prevTopicCount_d(k) > ACTIVE_THR) {
                    Kactive += 1;
                    ElogProb_d(k) = boost::math::digamma(
                        topicCount_d(k) + alphaEbeta(k));
                } else {
                    ElogProb_d(k) = digamma_alphaEbeta(k);
                }
            }
            // Need to pick at least nnzPerRow active topics for this doc.
            int k = 0;
            while (Kactive < nnzPerRow) {    
                if (prevTopicCount_d(k) < ACTIVE_THR) {
                    prevTopicCount_d(k) = ACTIVE_THR + 1e-9;
                    Kactive++;
                }
                k++;
            }
            */
        }
        //printf("iter=%d Kactive=%d\n", iter, Kactive);

        topicCount_d.fill(0);
        // Step over each data atom
        for (int n = 0; n < N; n++) {
            int m = n * nnzPerRow;
            int argmax_n = 0;
            double maxScore_n;
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                logScores_n(ka) = ElogProb_d(k) + ElogLik_d(n,k);
                if (ka == 0 || logScores_n(ka) > maxScore_n) {
                    maxScore_n = logScores_n(ka);
                    if (nnzPerRow == 1) {
                        argmax_n = k;
                    }
                }
            }
            /*
            int a = 0;
            for (int k = 0; k < K; k++) {
                if (prevTopicCount_d(k) > ACTIVE_THR) {
                    logScores_n(a) = ElogProb_d(k) + ElogLik_d(n,k);
                    if (a == 0 || logScores_n(a) > maxScore_n) {
                        maxScore_n = logScores_n(a);
                        if (nnzPerRow == 1) {
                            argmax_n = k;
                        }
                    }
                    a++;
                }
            }
            */
            if (nnzPerRow == 1) {
                spResp_data(m) = 1.0;
                spResp_colids(m) = argmax_n;
                // Update topicCount_d
                topicCount_d(argmax_n) += 1.0;

                if (doTrackELBO) {
                    totalLogSumResp += maxScore_n;
                }

            } else {
                // Find the top L entries in logScores_n
                // Copy current row over into a temp buffer
                std::copy(
                    logScores_n.data(),
                    logScores_n.data() + Kactive,
                    tempScores_n.data());
                // Sort the data in the temp buffer (in place)
                std:nth_element(
                    tempScores_n.data(),
                    tempScores_n.data() + Kactive - nnzPerRow,
                    tempScores_n.data() + Kactive);
                // Walk thru this row and find the top "nnzPerRow" positions
                double pivotScore = tempScores_n(Kactive - nnzPerRow);

                int nzk = m;
                double sumResp_n = 0.0;
                for (int ka = 0; ka < Kactive; ka++) {
                    if (logScores_n(ka) >= pivotScore) {
                        spResp_data(nzk) = \
                            exp(logScores_n(ka) - maxScore_n);
                        spResp_colids(nzk) = activeTopics_d(ka);
                        sumResp_n += spResp_data(nzk);
                        nzk += 1;                        
                    }
                }
                /*
                int a = 0;
                for (int k = 0; k < K; k++) {
                    if (prevTopicCount_d(k) > ACTIVE_THR) {
                        if (logScores_n(a) >= pivotScore) {
                            spResp_data(m + nzk) = \
                                exp(logScores_n(a) - maxScore_n);
                            spResp_colids(m + nzk) = k;
                            sumResp_n += spResp_data(m + nzk);
                            nzk += 1;
                        }
                        a += 1;
                    }
                }
                */
                for (int nzk = m; nzk < m + nnzPerRow; nzk++) {
                    spResp_data(nzk) /= sumResp_n;
                    topicCount_d(spResp_colids(nzk)) += \
                        spResp_data(nzk);
                }

                if (doTrackELBO) {
                    totalLogSumResp += maxScore_n + log(sumResp_n);
                }
            }

        }


        if (doTrackELBO) {
            elboVec(iter) = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            /*
            double elboV2 = calcELBOForSingleDoc_V2(
                alphaEbeta, topicCount_d, spResp_data, spResp_colids,
                ElogLik_d, K, N, nnzPerRow);
            
            double elboV1 = calcELBOForSingleDoc_V1(
                alphaEbeta, topicCount_d, ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            printf(" V1: %.6f\n V2: %.6f\n", elboV2, elboV1);
            */
            
        }
        //printf("END OF iter=%d  topicCount_d\n", iter);
        //for (int k = 0; k < K; k++) {
        //    printf("%7.2f ", topicCount_d(k));
        //}
        //printf("\n");
        // END ITERATION. Decide whether to quit early
        if (iter > 0 && iter % 5 == 0) {
            double absDiff_k = 0.0;
            maxDiff = 0.0;
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
            }
            /*
            for (int k = 0; k < K; k++) {
                absDiff_k = abs(prevTopicCount_d(k) - topicCount_d(k));
                if (absDiff_k > maxDiff) {
                    maxDiff = absDiff_k;
                }
            }
            */
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
    maxDiffVec(d) = maxDiff;
    numIterVec(d) = iter + 1;
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
        int initProbsToEbeta,
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

    for (int iter = 0; iter < nCoordAscentIterLP + initProbsToEbeta; iter++) {

        if (iter == 0 and initProbsToEbeta == 1) {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = log(alphaEbeta(k));
            }
        } else {
            for (int k = 0; k < K; k++) {
                ElogProb_d(k) = boost::math::digamma(
                    topicCount_d(k) + alphaEbeta(k));
            }
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
            if (maxDiff <= convThrLP) {
                break;
            }
        }
    }
}



