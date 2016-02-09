#include <math.h>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

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



struct Argsortable1DArray {
    double* xptr;
    int* iptr;
    int size;
    
    Argsortable1DArray(double* xptrIN, int sizeIN) {
        xptr = xptrIN;
        size = sizeIN;
        iptr = new int[size];
        resetIndices();
    }

    void resetIndices() {
        for (int i = 0; i < size; i++) {
            iptr[i] = i;
        }
    }

    bool operator()(int i, int j) {
        return xptr[i] < xptr[j];
    }

    void pprint() {
        for (int i = 0; i < size; i++) {
            printf("%4d:%7.1f ", iptr[i], xptr[i]);
        }
        printf("\n");
    }
};

/*
void findTopL_WITHSTRUCT(
    ExtArr1D_d scoreVec,
    ExtArr1D_d topLdata,
    ExtArr1D_i topLinds,
    int topL,
    int K)
{
    Arr1D_d indsVec (K);
    for (int k = 0; k < K; k++) {
        indsVec(k) = k;
    }

    // Sort the data in the temp buffer (in place)
    std::nth_element(
        indsVec.data(),
        indsVec.data() + K - topL,
        indsVec.data() + K,
        LessThanByIndex(tempScoreVec));
    
    for (int i = 0; i < topL; i++) {
        topLdata(i) = scoreVec(K - i);
        topLinds(i) = indsVec(K-i);
    }
}

void findTopL_WITHCOPY(
    ExtArr1D_d scoreVec,
    ExtArr1D_d topLdata,
    ExtArr1D_i topLinds,
    int topL,
    int K)
{
    Arr1D_d tempScoreVec (K);

    std::copy(
        scoreVec.data(),
        scoreVec.data() + K,
        tempScoreVec.data());

    // Sort the data in the temp buffer (in place)
    std::nth_element(
        tempScoreVec.data(),
        tempScoreVec.data() + K - topL,
        tempScoreVec.data() + K);
    
    // Walk thru this row and find the top "L" positions
    double pivotScore = tempScores_n(K - topL);

    int i = 0;
    for (int k = 0; k < K; k++) {
        if (scoreVec(k) >= pivotScore) {
            topLdata(i) = scoreVec(k);
            topLinds(i) = k;
            i += 1;
        }
    }
}
*/
int main(int argc, char* argv) {
    int K = 10;
    int topL = 4;
    Arr1D_d scoreVec = Arr1D_d::Random(K);

    for (int k = 0; k < K; k++) {
        printf("% 7.3f ", scoreVec(k));
    }
    printf("\n");

    Argsortable1DArray AV = Argsortable1DArray(scoreVec.data(), K);
    AV.pprint();

    //findTopL_WITHCOPY(scoreVec, topLdata, topLinds, K, topL);

    return 0;
}
