#include <iostream>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ======================================================== Func Declaration
// ========================================================

// Declare functions that need to be visible externally
extern "C" {
  void FwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int K,
    int T);

  void BwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * bwdMsgOUT,
    int K,
    int T);

  void SummaryAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * fwdMsgIN,
    double * bwdMsgIN,
    double * TransStateCountOUT,
    double * HtableOUT,
    int K,
    int T);
}



// ======================================================== Custom Type Defs
// ========================================================
// Simple names for array types
typedef Array<double, Dynamic, Dynamic, RowMajor> Arr2D;
typedef Array<double, 1, Dynamic, RowMajor> Arr1D;

// Simple names for array types with externally allocated memory
typedef Map<Arr2D> ExtArr2D;
typedef Map<Arr1D> ExtArr1D;



// ======================================================== Forward Algorithm
// ======================================================== 
void FwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * fwdMsgOUT,
    double * margPrObsOUT,
    int K,
    int T)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);

    // Prep output
    ExtArr2D fwdMsg (fwdMsgOUT, T, K);
    ExtArr1D margPrObs (margPrObsOUT, T);

    // Base case update for first time-step
    fwdMsg.row(0) = initPi * SoftEv.row(0);
    margPrObs(0) = fwdMsg.row(0).sum();
    fwdMsg.row(0) /= margPrObs(0);
    
    // Recursive update of timesteps 1, 2, ... T-1
    // Note: fwdMsg.row(t) is a *row vector* 
    //       so needs to be left-multiplied to square matrix transPi
    for (int t = 1; t < T; t++) {
        fwdMsg.row(t) = fwdMsg.row(t-1).matrix() * transPi.matrix();
        fwdMsg.row(t) *= SoftEv.row(t);
        margPrObs(t) = fwdMsg.row(t).sum();
        fwdMsg.row(t) /= margPrObs(t);
    }
}



// ======================================================== Backward Algorithm
// ======================================================== 
void BwdAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * bwdMsgOUT,
    int K,
    int T)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);
    ExtArr1D margPrObs (margPrObsIN, T);

    // Prep output
    ExtArr2D bMsg (bwdMsgOUT, T, K);

    // Base case update for last time-step
    bMsg.row(T-1).fill(1.0);
    
    // Recursive update of timesteps T-2, T-3, ... 3, 2, 1, 0
    // Note: bMsg.row(t) is a *row vector*
    //       so needs to be left-multiplied to square matrix transPi.T
    for (int t = T-2; t >= 0; t--) {
        bMsg.row(t) = (bMsg.row(t+1) * SoftEv.row(t+1)).matrix() \
                       * transPi.transpose().matrix();
        bMsg.row(t) /= margPrObs(t+1);
    }

}


// ======================================================== Summary Algorithm
// ======================================================== 
void SummaryAlg(
    double * initPiIN,
    double * transPiIN,
    double * SoftEvIN,
    double * margPrObsIN,
    double * fwdMsgIN,
    double * bwdMsgIN,
    double * TransStateCountOUT,
    double * HtableOUT,
    int K,
    int T)
{
    // Prep input
    ExtArr1D initPi (initPiIN, K);
    ExtArr2D transPi (transPiIN, K, K);
    ExtArr2D SoftEv (SoftEvIN, T, K);
    ExtArr1D margPrObs (margPrObsIN, T);
    ExtArr2D fwdMsg (fwdMsgIN, T, K);
    ExtArr2D bwdMsg (bwdMsgIN, T, K);

    // Prep output
    ExtArr2D TransStateCount (TransStateCountOUT, K, K);
    ExtArr2D Htable (HtableOUT, K, K);
  
    // Temporary KxK array for storing respPair at timestep t
    Arr2D respPair_t = ArrayXXd::Zero(K, K);
    Arr1D logrowwiseSum = ArrayXd::Zero(K);

    Arr2D epsArr = 1e-100 * ArrayXXd::Ones(K, K);

    for (int t = 1; t < T; t++) {
        // In Python, we want:
        // >>> respPair[t] = np.outer(fmsg[t-1], bmsg[t] * SoftEv[t])
        // >>> respPair[t] *= PiMat / margPrObs[t]
        respPair_t = fwdMsg.row(t-1).transpose().matrix() \
                      * (bwdMsg.row(t) * SoftEv.row(t)).matrix();
        respPair_t *= transPi;
        respPair_t /= margPrObs(t);


        // Aggregate pairwise transition counts
        TransStateCount += respPair_t;


        // Aggregate entropy in a KxK matrix

        // Step 1/3: Make numerically safe for logarithms
        // Each entry in respPair_t will be at least eps (1e-100)
        // Remember, cwiseMax only works with arrays, not scalars :(
        // https://forum.kde.org/viewtopic.php?f=74&t=98384
        respPair_t = respPair_t.cwiseMax(epsArr);

        // Step 2/3: Increment by rP log rP
        Htable += respPair_t * respPair_t.log();

        // Step 3/3: Decrement by rP log rP.rowwise().sum()
        // Remember, broadcasting with *= doesnt work
        // https://forum.kde.org/viewtopic.php?f=74&t=95629 
        // so we use a forloop instead
        logrowwiseSum = respPair_t.rowwise().sum();
        logrowwiseSum = logrowwiseSum.log();
        for (int k=0; k < K; k++) {
          respPair_t.col(k) *= logrowwiseSum;
        }
        Htable -= respPair_t; 

        /*
        printf("----------- t=%d\n", t);
        for (int j = 0; j < K; j++) {
          for (int k = 0; k < K; k++) {
            printf(" %.3f", respPair_t(j,k));
          }
          printf("\n");
        }
        */
    }
    Htable *= -1.0;
}
