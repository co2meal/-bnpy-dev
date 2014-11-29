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
