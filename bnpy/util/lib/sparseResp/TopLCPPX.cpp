
void main(int argc, char* argv) {
    K = 10;
    topL = 4;
    Arr1D_d scoreVec = Arr1D_d::Rand(topK);
    findTopL_WITHCOPY(scoreVec, topLdata, topLinds, K, topL);
}

struct LessThanByIndex {
    const double* xptr;

    LessThanByIndex(const double &xvec) {
        xptr = &xvec;
    }

    bool operator()(int i1, int i2) {
        return xptr(i1) < xptr(i2);
    }
};

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