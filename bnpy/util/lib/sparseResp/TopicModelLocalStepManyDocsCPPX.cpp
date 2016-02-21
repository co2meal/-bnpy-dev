// TopicModelLocalStepCPPX.cpp
// Define this symbol to enable runtime tests for allocations
#define EIGEN_RUNTIME_NO_MALLOC 

#include <math.h>
#include <time.h>
#include "Eigen/Dense"
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include "fastexp.h"
using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ======================================================== visible externally

extern "C" {
    void sparseLocalStepManyDocs_ActiveOnly(  
        double* alphaEbeta_IN,
        double* Eloglik_IN,
        double* word_count_IN,
        int* word_id_IN,
        int* doc_range_IN,
        int nnzPerRow,
        int Nall,
        int K,
        int D,
        int V,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int numRestarts,
        int* rAcceptVec_IN,
        int* rTrialVec_IN,
        int REVISE_FIRST,
        int REVISE_EVERY,
        int verbose
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


double calcElapsedTime(timespec start_time, timespec end_time) {
    double diffSec = (double) end_time.tv_sec - start_time.tv_sec;
    double diffNano = (double) end_time.tv_nsec - start_time.tv_nsec;
    return diffSec + diffNano / 1.0e9;
}

void precomputeTopLRespForEachVocabTerm(
        int nnzPerRow,
        int V,
        int K,
        ExtArr2D_d & Eloglik,
        ExtArr1D_d & alphaEbeta,
        Arr1D_d & ElogProb,
        Arr1D_d & logScores_n,
        Arr1D_d & tempScores_n,
        Arr1D_d & termResp_data,
        Arr1D_i & termResp_colids
        )
{
    ElogProb = alphaEbeta.log();
    for (int v = 0; v < V; v++) {
        int argmax_n = -1;
        double maxScore_n;
        double curScore;
        for (int k = 0; k < K; k++) {
            curScore = ElogProb(k) + Eloglik(v, k);
            if (k == 0 || curScore > maxScore_n) {
                maxScore_n = curScore;
                if (nnzPerRow == 1) {
                    argmax_n = k;
                }
            }
            logScores_n(k) = curScore;
        } // end loop over K
        if (nnzPerRow == 1) {
            termResp_data(v) = 1.0;
            termResp_colids(v) = argmax_n;
        } else {
            // Find the top L entries in logScores_n
            // Copy current row over into a temp buffer
            std::copy(
                logScores_n.data(),
                logScores_n.data() + K,
                tempScores_n.data());
            // Sort the data in the temp buffer (in place)
            std::nth_element(
                tempScores_n.data(),
                tempScores_n.data() + K - nnzPerRow,
                tempScores_n.data() + K);
            // Walk thru this row and find the top "nnzPerRow" positions
            double pivotScore = tempScores_n(K - nnzPerRow);
            double sumResp_n = 0.0;
            int termResp_start = v * nnzPerRow;
            int termResp_id = termResp_start;
            for (int k = 0; k < K; k++) {
                if (logScores_n(k) >= pivotScore) {
                    termResp_data(termResp_id) = \
                        fastexp(logScores_n(k) - maxScore_n);
                    termResp_colids(termResp_id) = k;
                    sumResp_n += termResp_data(termResp_id);
                    termResp_id += 1;                        
                }
            }
            assert(termResp_id - v * nnzPerRow == nnzPerRow);
            for (termResp_id = v * nnzPerRow;
                    termResp_id < (v+1) * nnzPerRow;
                    termResp_id++) {
                termResp_data(termResp_id) /= sumResp_n;
            }

        }
    }    
}

int updateActiveSetForDoc(
        int d,
        ExtArr2D_d & topicCount,
        Arr1D_i & activeTopics_d,
        Arr1D_i & spareActiveTopics_d,
        double ACTIVE_THR,
        int prevKactive,
        int nnzPerRow
        )
{
    int newKactive = 0;
    int ia = 0; // spare inactive topics
    for (int a = 0; a < prevKactive; a++) {
        int k = activeTopics_d(a);
        if (topicCount(d, k) > ACTIVE_THR) {
            activeTopics_d(newKactive) = k;
            newKactive += 1;
        } else if (newKactive < nnzPerRow - ia) {
            spareActiveTopics_d(ia) = k;
            ia += 1;
        }
    }

    int Kactive = newKactive;
    while (Kactive < nnzPerRow) {
        int k = spareActiveTopics_d(Kactive - newKactive);
        activeTopics_d(Kactive) = k;
        Kactive++;
    }
    return Kactive;
}

/*
 * Initialize activeTopics_d from provided topicCount
 * All K possible topics are available.
 */
int updateActiveSetForDocFromScratch(
        int d,
        ExtArr2D_d & topicCount,
        Arr1D_i & activeTopics_d,
        Arr1D_i & spareActiveTopics_d,
        double ACTIVE_THR,
        int K,
        int nnzPerRow
        )
{
    int newKactive = 0;
    int ia = 0; // spare inactive topics
    for (int k = 0; k < K; k++) {
        if (topicCount(d, k) > ACTIVE_THR) {
            activeTopics_d(newKactive) = k;
            newKactive += 1;
        } else if (newKactive < nnzPerRow - ia) {
            spareActiveTopics_d(ia) = k;
            ia += 1;
        }
    }

    int Kactive = newKactive;
    while (Kactive < nnzPerRow) {
        int k = spareActiveTopics_d(Kactive - newKactive);
        activeTopics_d(Kactive) = k;
        Kactive++;
    }
    return Kactive;
}

double calcELBOForDoc(
    int d,
    ExtArr1D_d & alphaEbeta,
    ExtArr2D_d & topicCount,
    Arr1D_d & ElogProb_d,
    Arr1D_i & activeTopics_d,
    double totalLogSumResp,
    double sum_gammalnalphaEbeta,
    int Kactive
    )
{
    double ELBO = sum_gammalnalphaEbeta;
    for (int ka = 0; ka < Kactive; ka++) {
        int k = activeTopics_d(ka);
        ELBO += (
            boost::math::lgamma(topicCount(d, k) + alphaEbeta(k))
            - boost::math::lgamma(alphaEbeta(k))
            - topicCount(d, k) * ElogProb_d(k)
            );
    }
    ELBO += totalLogSumResp;
    return ELBO;
}


double updateAssignmentsForDoc_ReviseActiveSet(  
    int d,
    int start_d,
    int N_d,
    int nnzPerRow,
    int Kactive,
    ExtArr1D_d alphaEbeta, 
    ExtArr2D_d Eloglik,
    ExtArr1D_d word_count,
    ExtArr1D_i word_id,
    ExtArr2D_d & topicCount,
    ExtArr1D_d & spResp_data,
    ExtArr1D_i & spResp_colids,
    Arr1D_i & activeTopics_d,
    Arr1D_d & ElogProb_d,
    Arr1D_d & logScores_n,
    Arr1D_d & tempScores_n,
    int initProbsToEbeta,
    int doTrackELBO
    )
{
    double totalLogSumResp = 0.0;

    // Update ElogProb_d for active topics
    if (initProbsToEbeta == 1) {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            ElogProb_d(k) = log(alphaEbeta(k));
        }
    }  else {
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            ElogProb_d(k) = boost::math::digamma(
                topicCount(d, k) + alphaEbeta(k));
        }
    }
    // RESET topicCounts for doc d
    topicCount.row(d).fill(0);

    // Update Resp_d for active topics
    // UPDATE assignments, obeying sparsity constraint
    for (int n = start_d; n < start_d + N_d; n++) {
        int spRind_dn_start = n * nnzPerRow;
        double w_ct = word_count(n);
        int w_id = word_id(n);
        int argmax_n = 0;
        double maxScore_n;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            logScores_n(ka) = ElogProb_d(k) + Eloglik(w_id, k);
            if (ka == 0 || logScores_n(ka) > maxScore_n) {
                maxScore_n = logScores_n(ka);
                if (nnzPerRow == 1) {
                    argmax_n = k;
                }
            }
        }
        if (nnzPerRow == 1) {
            spResp_data(spRind_dn_start) = 1.0;
            spResp_colids(spRind_dn_start) = argmax_n;
            topicCount(d, argmax_n) += w_ct;
            if (doTrackELBO) {
                totalLogSumResp += w_ct * maxScore_n;
            }
        } else {
            // Find the top L entries in logScores_n
            // Copy current row over into a temp buffer
            std::copy(
                logScores_n.data(),
                logScores_n.data() + Kactive,
                tempScores_n.data());
            // Sort the data in the temp buffer (in place)
            std::nth_element(
                tempScores_n.data(),
                tempScores_n.data() + Kactive - nnzPerRow,
                tempScores_n.data() + Kactive);
            // Walk thru this row and find the top "nnzPerRow" positions
            double pivotScore = tempScores_n(Kactive - nnzPerRow);

            int spRind_dn = spRind_dn_start;
            double sumResp_n = 0.0;
            for (int ka = 0; ka < Kactive; ka++) {
                if (logScores_n(ka) >= pivotScore) {
                    spResp_data(spRind_dn) = \
                        fastexp(logScores_n(ka) - maxScore_n);
                    spResp_colids(spRind_dn) = activeTopics_d(ka);
                    sumResp_n += spResp_data(spRind_dn);
                    spRind_dn += 1;                        
                }
            }
            assert(spRind_dn - spRind_dn_start == nnzPerRow);
            for (spRind_dn = spRind_dn_start;
                    spRind_dn < spRind_dn_start + nnzPerRow; spRind_dn++) {
                spResp_data(spRind_dn) /= sumResp_n;
                topicCount(d, spResp_colids(spRind_dn)) += \
                    w_ct * spResp_data(spRind_dn);
            }
            if (doTrackELBO) {
                totalLogSumResp += w_ct * (maxScore_n + log(sumResp_n));
            }
        } // end if statement branch for nnz > 1
    } // end for loop over tokens in this doc
    return totalLogSumResp;
}


/* Update spResp_data values, keeping spResp_colids fixed.
 *
 */
void updateAssignmentsForDoc_FixPerTokenActiveSet(  
    int d,
    int start_d,
    int N_d,
    int nnzPerRow,
    int Kactive,
    ExtArr1D_d alphaEbeta, 
    ExtArr2D_d Eloglik,
    ExtArr1D_d word_count,
    ExtArr1D_i word_id,
    ExtArr2D_d & topicCount,
    ExtArr1D_d & spResp_data,
    ExtArr1D_i & spResp_colids,
    Arr1D_i & activeTopics_d,
    Arr1D_d & ElogProb_d,
    Arr1D_d & logScores_n
    )
{
    assert(nnzPerRow > 1);
    //timespec start_time, end_time;
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    // Update ElogProb_d for active topics
    for (int ka = 0; ka < Kactive; ka++) {
        int k = activeTopics_d(ka);
        ElogProb_d(k) = boost::math::digamma(
            topicCount(d, k) + alphaEbeta(k));
    }
    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    //double timeSpent_Digamma = calcElapsedTime(start_time, end_time);

    // RESET topicCounts for doc d
    topicCount.row(d).fill(0);

    //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
    // Update Resp_d for active topics
    for (int n = start_d; n < start_d + N_d; n++) {
        int spRind_dn_start = n * nnzPerRow;
        double w_ct = word_count(n);
        int w_id = word_id(n);
        double maxScore_n;
        for (int ka = 0; ka < nnzPerRow; ka++) {
            int k = spResp_colids(spRind_dn_start + ka);
            logScores_n(ka) = ElogProb_d(k) + Eloglik(w_id, k);
            if (ka == 0 || logScores_n(ka) > maxScore_n) {
                maxScore_n = logScores_n(ka);
            }
        }

        double sumResp_n = 0.0;
        for (int ka = 0; ka < nnzPerRow; ka++) {
            spResp_data(spRind_dn_start + ka) = \
                fastexp(logScores_n(ka) - maxScore_n);
            sumResp_n += spResp_data(spRind_dn_start + ka);
        }

        for (int ka = 0; ka < nnzPerRow; ka++) {
            spResp_data(spRind_dn_start + ka) /= sumResp_n;
            topicCount(d, spResp_colids(spRind_dn_start + ka)) += \
                w_ct * spResp_data(spRind_dn_start + ka);
        }

    } // end for loop over tokens in this doc
    /*clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
    double timeSpent_Resp = calcElapsedTime(start_time, end_time);
    logScores_n(0) = timeSpent_Digamma;
    logScores_n(1) = timeSpent_Resp;
    */
}


double tryRestartsForDoc(
    int d,
    int start_d,
    int N_d,
    int nnzPerRow,
    int Kactive,
    ExtArr1D_d alphaEbeta, 
    ExtArr2D_d Eloglik,
    ExtArr1D_d word_count,
    ExtArr1D_i word_id,
    ExtArr2D_d & topicCount,
    ExtArr1D_d & spResp_data,
    ExtArr1D_i & spResp_colids,
    Arr1D_i & activeTopics_d,
    Arr1D_d & prevTopicCount_d,
    Arr1D_d & ElogProb_d,
    Arr1D_d & logScores_n,
    Arr1D_d & tempScores_n,
    ExtArr1D_i & rAcceptVec,
    ExtArr1D_i & rTrialVec,
    int numRestarts,
    double sum_gammalnalphaEbeta,
    int verbose
    )
{
    // Find active topics eligible for sparse restart
    int nAccept = 0;
    int nTrial = 0;
    double SMALL_THR = 1e-15;
    double curELBO = 0.0;
    double totalLogSumResp = 0.0;
    prevTopicCount_d.fill(0);
    if (verbose > 1) {
        printf("SPARSE RESTARTS at doc %d!!\n", d);
    }
    for (int riter = 0; riter < numRestarts; riter++) {
        if (riter == 0) {
            totalLogSumResp = updateAssignmentsForDoc_ReviseActiveSet(
                d, start_d, N_d, nnzPerRow, Kactive,
                alphaEbeta, Eloglik, word_count, word_id, 
                topicCount, spResp_data, spResp_colids,
                activeTopics_d, ElogProb_d, logScores_n, tempScores_n,
                0, 1);
            // ELBO for current configuration
            curELBO = calcELBOForDoc(d, alphaEbeta, topicCount,
                ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
            // Remember the best-known topic-count vector!
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                prevTopicCount_d(k) = topicCount(d, k);
            }
        }
        // SEARCH FOR SMALLEST TOPIC HAVE NOT YET TRIED YET
        int numAboveThr = 0;
        double minVal = N_d + 1.0; // Will never be a value in topicCount
        int minLoc = 0;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            if (topicCount(d, k) > SMALL_THR) {
                numAboveThr += 1;
                if (topicCount(d, k) < minVal) {
                    minVal = topicCount(d, k);
                    minLoc = k;
                }
            }
        }
        if (numAboveThr == 1) {
            break;
        }
        if (verbose > 1) {
            printf("START: best known counts. ELBO=%.5e \n", curELBO);
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount(d, k));
            }
            printf("\n");
        }
        SMALL_THR = minVal;
        // Force chosen topic to zero
        topicCount(d, minLoc) = 0.0;
        if (verbose > 1) {        
            printf(
                "RESTART: Set index %d to zero (%d left)\n", 
                minLoc, numAboveThr - 1);
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount(d, k));
            }
            printf("\n");
        }
        // Run inference forward from forced new location
        int NSTEP = 2;
        for (int step = 0; step < NSTEP; step++) { 
            totalLogSumResp = updateAssignmentsForDoc_ReviseActiveSet(
                d, start_d, N_d, nnzPerRow, Kactive,
                alphaEbeta, Eloglik, word_count, word_id, 
                topicCount, spResp_data, spResp_colids,
                activeTopics_d, ElogProb_d, logScores_n, tempScores_n,
                0, step == NSTEP - 1);
        }
        // If the change is small, abandon current proposal
        double propELBO;
        if (abs(prevTopicCount_d(minLoc) - topicCount(d, minLoc)) < 1e-5) {
            propELBO = curELBO;
        } else {
            propELBO = calcELBOForDoc(d, alphaEbeta, topicCount,
                ElogProb_d, activeTopics_d,
                totalLogSumResp, sum_gammalnalphaEbeta, Kactive);
        }
        if (verbose > 1) {
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                printf("%02d:%06.2f ", k, topicCount(d, k));
            }
            printf("\n");
            printf("propELBO % .6e\n", propELBO);
            printf(" curELBO % .6e\n", curELBO);
            if (propELBO > curELBO) {
                printf("beforeCount: %.6f\n", prevTopicCount_d(minLoc));
                printf(" afterCount: %.6f\n", topicCount(d, minLoc));
                printf("gainELBO % .6e ACCEPTED \n", propELBO - curELBO);
            } else {
                printf("gainELBO % .6e rejected \n", propELBO - curELBO);
            }
        }
        // If accepted, set current best doc-topic counts to latest proposal
        // Otherwise, reset the starting point for the next proposal.
        if (propELBO > curELBO) {
            curELBO = propELBO;
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                prevTopicCount_d(k) = topicCount(d, k);
            }
            nAccept += 1;
        } else {
            for (int ka = 0; ka < Kactive; ka++) {
                int k = activeTopics_d(ka);
                topicCount(d, k) = prevTopicCount_d(k);
            }
        }
        nTrial += 1;
    } // end loop over restarts


    if (numRestarts > 0) {
        rAcceptVec(0) += nAccept;
        rTrialVec(0) += nTrial;
        for (int ka = 0; ka < Kactive; ka++) {
            int k = activeTopics_d(ka);
            topicCount(d, k) = prevTopicCount_d(k);
        }
        // Final update! Make sure spResp reflects best topicCounts found
        updateAssignmentsForDoc_ReviseActiveSet(
            d, start_d, N_d, nnzPerRow, Kactive,
            alphaEbeta, Eloglik, word_count, word_id, 
            topicCount, spResp_data, spResp_colids,
            activeTopics_d, ElogProb_d, logScores_n, tempScores_n,
            0, 0);
    } // end if to synchronize at best known assignments

    return curELBO;
}

void sparseLocalStepManyDocs_ActiveOnly(
        double* alphaEbeta_IN,
        double* Eloglik_IN,
        double* word_count_IN,
        int* word_id_IN,
        int* doc_range_IN,
        int nnzPerRow,
        int Nall,
        int K,
        int D,
        int V,
        int nCoordAscentIterLP,
        double convThrLP,
        int initProbsToEbeta,
        double* topicCount_OUT,
        double* spResp_data_OUT,
        int* spResp_colids_OUT,
        int* numIterVec_OUT,
        double* maxDiffVec_OUT,
        int numRestarts,
        int* rAcceptVec_IN,
        int* rTrialVec_IN,
        int REVISE_FIRST,
        int REVISE_EVERY,
        int verbose
        )
{
    /*timespec start_time, end_time;
    double timeSpent_FindActive = 0.0;
    double timeSpent_Assign = 0.0;
    double timeSpent_AssignFixed = 0.0;
    double timeSpent_Restart = 0.0;
    double timeSpent_AssignFixed_Digamma = 0.0;
    double timeSpent_AssignFixed_Resp = 0.0;
    */
    nCoordAscentIterLP = max(nCoordAscentIterLP, 0);
    if (initProbsToEbeta == 1) {
        nCoordAscentIterLP += 1;
    }
    nCoordAscentIterLP = max(nCoordAscentIterLP, 1);

    if (nnzPerRow == 1) {
        REVISE_EVERY = 1;
    } else {
        REVISE_EVERY = max(REVISE_EVERY, 1);    
    }
    REVISE_FIRST = max(REVISE_FIRST, 0);

    int CHECK_EVERY = 5; // Check convergence every X iterations.

    // Allocate temporary storage
    Arr1D_d ElogProb_d (K);
    Arr1D_d prevTopicCount_d (K);
    Arr1D_d logScores_n (K);
    Arr1D_d tempScores_n (K);
    Arr1D_i activeTopics_d (K);
    Arr1D_i spareActiveTopics_d (nnzPerRow);
    
    Arr1D_d termResp_data;
    Arr1D_i termResp_colids;

    if (initProbsToEbeta == 2) {
        termResp_data = Arr1D_d::Zero(V * nnzPerRow);
        termResp_colids = Arr1D_i::Zero(V * nnzPerRow);   
    }

    // Disable all further memory allocation.
    // We do this to verify that we have no memory leaks!
    internal::set_is_malloc_allowed(false);

    // Unpack input arrays
    ExtArr1D_d alphaEbeta (alphaEbeta_IN, K);
    ExtArr2D_d Eloglik (Eloglik_IN, V, K);

    ExtArr1D_d word_count (word_count_IN, Nall);
    ExtArr1D_i word_id (word_id_IN, Nall);
    ExtArr1D_i doc_range (doc_range_IN, D+1);

    // Unpack output arrays
    ExtArr1D_d spResp_data (spResp_data_OUT, Nall * nnzPerRow);
    ExtArr1D_i spResp_colids (spResp_colids_OUT, Nall * nnzPerRow);
    ExtArr2D_d topicCount (topicCount_OUT, D, K);

    ExtArr1D_i numIterVec (numIterVec_OUT, D);
    ExtArr1D_d maxDiffVec (maxDiffVec_OUT, D);

    ExtArr1D_i rAcceptVec (rAcceptVec_IN, 1);
    ExtArr1D_i rTrialVec (rTrialVec_IN, 1);
    
    // Compute quantity used in sparse restart ELBO computation
    double sum_gammalnalphaEbeta = 0.0;
    if (numRestarts > 0) {
        for (int k = 0; k < K; k++) {
            sum_gammalnalphaEbeta += boost::math::lgamma(alphaEbeta(k));
        }
    }

    if (initProbsToEbeta == 2) {
        precomputeTopLRespForEachVocabTerm(
            nnzPerRow, V, K,
            Eloglik,
            alphaEbeta,
            ElogProb_d,
            logScores_n,
            tempScores_n,
            termResp_data,
            termResp_colids);
        if (verbose) {
            printf("Precalculated Topics-by-term\n");
            for (int v = 0; v < 5; v++) {
                printf("term %06d ", v);
                for (int vid = v * nnzPerRow; vid < (v+1) * nnzPerRow; vid++) {
                    printf("%03d:%.2f ",
                        termResp_colids(vid), termResp_data(vid));
                }
                printf("\n");
            }
            for (int v = V-5; v < V; v++) {
                printf("term %06d ", v);
                for (int vid = v * nnzPerRow; vid < (v+1) * nnzPerRow; vid++) {
                    printf("%03d:%.2f ",
                        termResp_colids(vid), termResp_data(vid));
                }
                printf("\n");
            }
        }
    }

    // Visit each document and update its spResp (and corresponding topicCount)
    for (int d = 0; d < D; d++) {
        int start_d = doc_range(d);
        int N_d = doc_range(d+1) - doc_range(d);

        prevTopicCount_d.fill(0);
        double maxDiff = N_d;
        int iter = 0;
        double ACTIVE_THR = 1e-9;
        int doReviseActiveSet = 1;
        int Kactive = K;
        // Initialize activeTopics_d and topicCount_d    
        if (initProbsToEbeta == 2) {
            // using the precomputed per-token resp values!
            for (int n = start_d; n < start_d + N_d; n++) {
                int wid = word_id(n);
                for (int a = wid * nnzPerRow; a < (wid+1) * nnzPerRow; a++) {
                    int k = termResp_colids(a);
                    topicCount(d, k) += word_count(n) * termResp_data(a);
                }
            }
            Kactive = updateActiveSetForDocFromScratch(
                d,
                topicCount,
                activeTopics_d,
                spareActiveTopics_d,
                ACTIVE_THR,
                K, nnzPerRow);
        } else {
            // Initialize active set to ALL topics    
            Kactive = K;
            for (int k = 0; k < K; k++) {
                activeTopics_d(k) = k;
            }
        }

        for (iter = 0; iter < nCoordAscentIterLP; iter++) {
            //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
            // DETERMINE CURRENT ACTIVE SET
            if (nnzPerRow > 1 && Kactive <= nnzPerRow) {
                // Set of active docs is already as small as can be,
                // so nothing to gain from revising
                doReviseActiveSet = 0;
            } else if (iter < REVISE_FIRST || (iter - 1) % REVISE_EVERY == 0) {
                doReviseActiveSet = 1;
            } else {
                doReviseActiveSet = 0;
            }
            if (iter > 0 && doReviseActiveSet) {
                Kactive = updateActiveSetForDoc(
                    d,
                    topicCount,
                    activeTopics_d,
                    spareActiveTopics_d,
                    ACTIVE_THR,
                    Kactive, nnzPerRow);
            }
            assert(Kactive >= nnzPerRow);
            assert(Kactive <= K);
            //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
            //timeSpent_FindActive += calcElapsedTime(start_time, end_time);

            // COMPUTE ASSIGNMENTS FOR CURRENT ACTIVE SET
            if (doReviseActiveSet) {
                //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
                updateAssignmentsForDoc_ReviseActiveSet(
                    d, start_d, N_d, nnzPerRow, Kactive,
                    alphaEbeta,
                    Eloglik,
                    word_count,
                    word_id,
                    topicCount,
                    spResp_data,
                    spResp_colids,
                    activeTopics_d,
                    ElogProb_d,
                    logScores_n,
                    tempScores_n,
                    (initProbsToEbeta == 1) && (iter == 0),
                    0);
                //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
                //timeSpent_Assign += calcElapsedTime(start_time, end_time);
            } else {
                //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
                updateAssignmentsForDoc_FixPerTokenActiveSet(
                    d, start_d, N_d, nnzPerRow, Kactive,
                    alphaEbeta,
                    Eloglik,
                    word_count,
                    word_id,
                    topicCount,
                    spResp_data,
                    spResp_colids,
                    activeTopics_d,
                    ElogProb_d,
                    logScores_n
                    );
                /*clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
                timeSpent_AssignFixed += calcElapsedTime(start_time, end_time);
                timeSpent_AssignFixed_Digamma += logScores_n(0);
                timeSpent_AssignFixed_Resp += logScores_n(1);
                */
            }

            // END ITERATION. Decide whether to quit early
            // Find maximum difference from previous doc-topic count
            if (iter % CHECK_EVERY == 0 && iter > 0) {
                double absDiff_k = 0.0;
                maxDiff = 0.0;
                for (int ka = 0; ka < Kactive; ka++) {
                    int k = activeTopics_d(ka);
                    absDiff_k = abs(prevTopicCount_d(k) - topicCount(d, k));
                    if (absDiff_k > maxDiff) {
                        maxDiff = absDiff_k;
                    }
                }
            }
            if (verbose > 1) {
                printf("iter %3d doRevise %d Kactive %4d maxDiff %11.6f\n",
                    iter, doReviseActiveSet, Kactive, maxDiff);
                printf("  ");
                for (int ka = 0; ka < Kactive; ka++) {
                    int k = activeTopics_d(ka);
                    printf("%d:%5.1f ", k, topicCount(d,k));
                }
                printf("\n");
            }
            if (maxDiff <= convThrLP) {
                if (verbose > 1) {
                    printf("EARLY EXIT! maxDiff < %11.6f\n", convThrLP);
                }
                break;
            }

            if (iter % CHECK_EVERY == CHECK_EVERY - 1) {
                // Make sure prevTopicCount vector is updated
                for (int ka = 0; ka < Kactive; ka++) {
                    int k = activeTopics_d(ka);
                    prevTopicCount_d(k) = topicCount(d, k);
                }
            }

        } // end loop over iterations at doc d
        maxDiffVec(d) = maxDiff;
        numIterVec(d) = iter; // iter will already be +1'd by last iter of loop

        if (numRestarts > 0) {
            //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &start_time);
            tryRestartsForDoc(
                d, start_d, N_d, nnzPerRow, Kactive,
                alphaEbeta,
                Eloglik,
                word_count,
                word_id,
                topicCount,
                spResp_data,
                spResp_colids,
                activeTopics_d,
                prevTopicCount_d,
                ElogProb_d,
                logScores_n,
                tempScores_n,
                rAcceptVec,
                rTrialVec,
                numRestarts,
                sum_gammalnalphaEbeta,
                verbose
                );
           //clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &end_time);
           //timeSpent_Restart += calcElapsedTime(start_time, end_time);

        }
    } // end loop over documents
    internal::set_is_malloc_allowed(true);

    if (verbose > 0) {
        printf(
            "Sparse Restarts: %d/%d accepted\n", rAcceptVec(0), rTrialVec(0));
    }

    /*
    double timeSpent_total = \
        timeSpent_FindActive + timeSpent_Assign + \
        timeSpent_AssignFixed + timeSpent_Restart;

    printf("FindActive   %8.3f sec  %.3f %%\n", timeSpent_FindActive,
        timeSpent_FindActive / timeSpent_total);
    printf("Assign:      %8.3f sec  %.3f %%\n", timeSpent_Assign,
        timeSpent_Assign / timeSpent_total);
    printf("AssignFixed: %8.3f sec  %.3f %%\n", timeSpent_AssignFixed,
        timeSpent_AssignFixed / timeSpent_total);
    printf("  -Digamma:  %8.3f sec\n", timeSpent_AssignFixed_Digamma);
    printf("  -Resp:     %8.3f sec\n", timeSpent_AssignFixed_Resp);
    printf("Restarts:    %8.3f sec  %.3f %%\n", timeSpent_Restart,
        timeSpent_Restart / timeSpent_total);
    printf("TOTAL:       %8.3f sec\n", timeSpent_total);
    */
}

