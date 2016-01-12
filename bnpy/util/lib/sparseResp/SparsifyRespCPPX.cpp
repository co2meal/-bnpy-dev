/* SparseRespCPPX.cpp
Fast implementation of sparsifyResp
*/
#define EIGEN_RUNTIME_NO_MALLOC // Define this symbol to enable runtime tests for allocations

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

    void sparsifyLogResp(
        double* logResp_IN,
        int nnzPerRow,
        int N,
        int K,
        double* spR_data_OUT,
        int* spR_colids_OUT
        );

    void calcRlogR_withSparseRespCSR(
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int K,
        int N,
        int nnzPerRow,
        double* Hvec_OUT
        );

    void calcRXXT_withSparseRespCSR(
        double* X_IN,
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int D,
        int K,
        int N,
        int nnzPerRow,
        double* stat_RXX_OUT
        );

    void calcRXX_withSparseRespCSR(
        double* X_IN,
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int D,
        int K,
        int N,
        int nnzPerRow,
        double* stat_RXX_OUT
        );

    void calcRXX_withSparseRespCSC(
        double* X_IN,
        double* spR_data_IN,
        int* spR_rowids_IN,
        int* spR_colptr_IN,
        int D,
        int K,
        int L,
        int N,
        double* stat_RXX_OUT
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

/*
Careful to avoid overflow in exp(logResp) by always subtracting maximum value.
 */
void sparsifyLogResp(
        double* logResp_IN,
        int nnzPerRow,
        int N,
        int K,
        double* spR_data_OUT,
        int* spR_colids_OUT)
{
    ExtArr2D_d logResp (logResp_IN, N, K);
    ExtArr1D_d spR_data (spR_data_OUT, N * nnzPerRow);
    ExtArr1D_i spR_colids (spR_colids_OUT, N * nnzPerRow);
    VectorXd curRow (K);
    for (int n = 0; n < N; n++) {
        // Copy current row over into a temp buffer
        std::copy(logResp.data() + (n * K),
                  logResp.data() + ((n+1) * K),
                  curRow.data());
        // Sort the data in the temp buffer (in place)
        std:nth_element(curRow.data(),
                        curRow.data() + K - nnzPerRow,
                        curRow.data() + K);

        // Walk through original data and find the top "nnzPerRow" positions
        double maxlogResp_n = curRow(K - 1);
        double pivot = curRow(K - nnzPerRow);
        int nzk = 0;
        int M = n * nnzPerRow;
        for (int k = 0; k < K; k++) {
            if (logResp(n,k) >= pivot) {
                spR_data(M + nzk) = logResp(n,k);
                spR_colids(M + nzk) = k;
                nzk += 1;
                if (logResp(n,k) > maxlogResp_n) {
                    maxlogResp_n = logResp(n,k);
                }
            }
        }
        //std::assert(nzk == nnzPerRow);

        // Compute exp of each of the non-zero values in row n
        double rowsum = 0.0;
        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            int m = M + nzk;
            spR_data(m) = exp(spR_data(m) - maxlogResp_n);
            rowsum += spR_data(m);
        }
        // Force all non-zero resp values for row n to sum to one.
        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            spR_data(M + nzk) /= rowsum;
        }
    }
    
}

void calcRlogR_withSparseRespCSR(
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int K,
        int N,
        int nnzPerRow,
        double* Hvec_OUT)
{
    ExtArr1D_d spR_data (spR_data_IN, N * nnzPerRow);
    ExtArr1D_i spR_colids (spR_colids_IN, N * nnzPerRow);
    ExtArr1D_i spR_rowptr (spR_rowptr_IN, K+1);
    ExtMat1D_d Hvec (Hvec_OUT, K);
    for (int n = 0; n < N; n++) {
        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            int m = n * nnzPerRow + nzk;
            if (spR_data(m) > 1e-20) {
                Hvec(spR_colids[m]) -= spR_data(m) * log(spR_data(m));
            }
        }
    }
}


void calcRXXT_withSparseRespCSR(
        double* X_IN,
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int D,
        int K,
        int N,
        int nnzPerRow,
        double* stat_RXX_OUT)
{
    ExtMat2D_d X (X_IN, N, D);
    ExtArr1D_d spR_data (spR_data_IN, N * nnzPerRow);
    ExtArr1D_i spR_colids (spR_colids_IN, N * nnzPerRow);
    ExtArr1D_i spR_rowptr (spR_rowptr_IN, K+1);
    ExtArr2D_d stat_RXX (stat_RXX_OUT, K, D * D);

    // Create storage for holding the outer-product of each data item
    // Using the Matrix type (not the Array type) avoids temporary allocation
    Mat2D_d xxT = Mat2D_d::Zero(D, D);
    Map<Arr1D_d> xxTvec (xxT.data(), xxT.size());
    //internal::set_is_malloc_allowed(false);
    //Arr2D_d xxT = Arr2D_d::Zero(D, D);
    //internal::set_is_malloc_allowed(true);

    for (int n = 0; n < N; n++) {
        // Compute outer-product
        // using noalias avoids any additional memory allocation
        xxT.noalias() = X.row(n).transpose() * X.row(n);
        //internal::set_is_malloc_allowed(false);
        //xxT = X.row(n).matrix().transpose() * X.row(n).matrix();
        //internal::set_is_malloc_allowed(true);

        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            int l = n * nnzPerRow + nzk;
            stat_RXX.row(spR_colids(l)) += spR_data(l) * xxTvec.array();
        }
    }
}

void calcRXX_withSparseRespCSR(
        double* X_IN,
        double* spR_data_IN,
        int* spR_colids_IN,
        int* spR_rowptr_IN,
        int D,
        int K,
        int N,
        int nnzPerRow,
        double* stat_RXX_OUT)
{
    ExtArr2D_d X (X_IN, N, D);
    ExtArr1D_d spR_data (spR_data_IN, N * nnzPerRow);
    ExtArr1D_i spR_colids (spR_colids_IN, N * nnzPerRow);
    ExtArr1D_i spR_rowptr (spR_rowptr_IN, K+1);
    ExtArr2D_d stat_RXX (stat_RXX_OUT, K, D);
    Arr1D_d xsq;
    for (int n = 0; n < N; n++) {
        xsq = X.row(n).square();
        for (int nzk = 0; nzk < nnzPerRow; nzk++) {
            int l = n * nnzPerRow + nzk;
            stat_RXX.row(spR_colids(l)) += \
                spR_data(l) * xsq;
        }
    }
}

void calcRXX_withSparseRespCSC(
        double* X_IN,
        double* spR_data_IN,
        int* spR_rowids_IN,
        int* spR_colptr_IN,
        int D,
        int K,
        int L,
        int N,
        double* stat_RXX_OUT)
{
    ExtArr2D_d X (X_IN, N, D);
    ExtArr1D_d spR_data (spR_data_IN, L);
    ExtArr1D_i spR_rowids (spR_rowids_IN, L);
    ExtArr1D_i spR_colptr (spR_colptr_IN, K+1);
    ExtArr2D_d stat_RXX (stat_RXX_OUT, K, D);
    for (int k = 0; k < K; k++) {
        for (int n = spR_colptr[k]; n < spR_colptr[k+1]; n++) {
            stat_RXX.row(k) += spR_data(n) * X.row(spR_rowids(n)).square();
        }
    }
}

