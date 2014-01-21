#!/bin/sh

export BNPYDATADIR=/data/NIPS/

ModelArgs=" --alpha0 5 --gamma 0.5"
AlgArgs=" --K 100 --nLap 3 --nCoordAscentItersLP 10 --doMemoizeLocalParams 0"
BirthArgs=" --birthPerLap 1 --minTargetSize 50 --maxTargetSize 300 --nFreshLap 25"
MergeArgs=" --mergePerLap 1000 --version 1 --doUpdateAllComps 0"

CMD="python RunWithProfiler.py NIPSCorpus HDPModel Mult moVB --moves birth,merge $ModelArgs $AlgArgs $BirthArgs $MergeArgs"

echo "Running CMD......."
echo $CMD
echo ".................."
$CMD 2>&1 | grep -v KeyError
