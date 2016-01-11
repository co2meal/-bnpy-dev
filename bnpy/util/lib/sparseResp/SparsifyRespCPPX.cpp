/* SparseRespCPPX.cpp
Fast implementation of sparsifyResp
*/
#include <math.h>
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

// ======================================================== Declare funcs
// ========================================================  visible externally

extern "C" {
    void sparsifyResp(
        double* Resp_IN,
        int nnzPerRow,
        int N,
        int K,
        double* spR_data_OUT,
        int* spR_colids_OUT
        );
}

// ======================================================== Custom Type Defs
// ========================================================
// Simple names for array types
typedef Array<double, Dynamic, Dynamic, RowMajor> Arr2D_d;
typedef Array<double, 1, Dynamic, RowMajor> Arr1D_d;
typedef Array<int, 1, Dynamic, RowMajor> Arr1D_i;

// Simple names for array types with externally allocated memory
typedef Map<Arr2D_d> ExtArr2D_d;
typedef Map<Arr1D_d> ExtArr1D_d;
typedef Map<Arr1D_i> ExtArr1D_i;

void sparsifyResp(
        double* Resp_IN,
        int nnzPerRow,
        int N,
        int K,
        double* spR_data_OUT,
        int* spR_colids_OUT)
{
    ExtArr2D_d Resp (Resp_IN, N, K);
    ExtArr1D_d spR_data (spR_data_OUT, N * nnzPerRow);
    ExtArr1D_i spR_colids (spR_colids_OUT, N * nnzPerRow);

    VectorXd curRow (K);

    
    for (int n = 0; n < N; n++) {
        // Copy current row over into a temp buffer
        std::copy(Resp.data() + (n * K),
                  Resp.data() + ((n+1) * K),
                  curRow.data());
        // Sort the data in the temp buffer (in place)
        std:nth_element(curRow.data(),
                        curRow.data() + K - nnzPerRow,
                        curRow.data() + K);

        // Walk through original data and find the top K positions
        double pivot = curRow(K - nnzPerRow);
        double rowsum = 0.0;
        int nzk = 0;
        for (int k = 0; k < K; k++) {
            if (Resp(n,k) >= pivot) {
                spR_data(n * nnzPerRow + nzk) = Resp(n,k);
                spR_colids(n * nnzPerRow + nzk) = k;
                rowsum += Resp(n,k);
                nzk += 1;
            }
        }

        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            spR_data(n * nnzPerRow + nzk) /= rowsum;
        }
    }
    
}
