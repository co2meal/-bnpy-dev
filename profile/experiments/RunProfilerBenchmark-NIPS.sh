#!/bin/sh
source experiments/SetupEnvBNPY-NIPS.sh
DataName="NIPSCorpus"
BASEDIR="reports/$DataName-timetrials/"

for AllocModel in HDPModel HDPSoft2Hard
do

rm -f "$BASEDIR/$AllocModel-results.txt"

for mergePerLap in 0 100
do

for nBatch in 10
do

for K in 5
do

for nThread in 2 4 8
do
export OMP_NUM_THREADS=$nThread

AlgArgs=" --K $K --nLap 1 --nCoordAscentLP 10 --doMemoizeLocalParams 0 --doFullPassBeforeMstep 0 --nBatch $nBatch"

ModelArgs=" --alpha0 5 --gamma 0.5"

if [[ $mergePerLap == "0" ]]; then
  MergeArgs=""
else
  MergeArgs=" --moves merge --doAllPairs 0 --mergePerLap $mergePerLap --doUpdateAllComps 0"
fi

JOBNAME="$AllocModel-merge$mergePerLap-K$K-nThread$nThread"

CMD="python RunWithProfiler.py $DataName $AllocModel Mult moVB --jobname $JOBNAME $ModelArgs $AlgArgs $MergeArgs"


OUTDIR="reports/$DataName-timetrials/$JOBNAME"

rm -rf $OUTDIR/*.txt
rm -rf $OUTDIR/*/*.html
mkdir -p $OUTDIR

echo "Running CMD..........................   $JOBNAME"
echo $CMD

$CMD 2>&1 | grep -v KeyError | grep -v [DONE] | grep -v bnpy > $OUTDIR/transcript.txt

lastLineStatus=`tail -n1 "$OUTDIR/transcript.txt"`

if [[ $lastLineStatus == *done* ]]; then
  echo "[DONE] confirmed. Status: $lastLineStatus. bnpy time: $elapsedTimeFromBnpy. profiler time: $elapsedTimeFromProf."
else
  echo "[ERROR] Status: "$lastLineStatus
  exit
fi

# Move profiler HTML output to desired directory
mv reports/MyProfile/* $OUTDIR

# Measure elapsed time from bnpy transcript
lapTimingLine=`tail -n2 "$OUTDIR/transcript.txt" | head -n1`
elapsedTimeFromBnpy=`echo $lapTimingLine \
  | cut -d' ' -f 3 \
  | xargs printf %.2f`

# Measure elapsed time from profiler output html
elapsedTimeFromProf=`grep -b3 "Run.py-run.html" $OUTDIR/index.html \
                      | tail -n1 |  cut -d'>' -f2 | cut -d' ' -f1`

echo $elapsedTimeFromBnpy > $OUTDIR/elapsedtime-bnpy.txt
echo $elapsedTimeFromProf > $OUTDIR/elapsedtime-prof.txt
echo "$K $mergePerLap $nThread $elapsedTimeFromBnpy $elapsedTimeFromProf" \
      >> "$BASEDIR/$AllocModel-results.txt"

echo "..............................................................."

done
done
done
done

echo "K mergePerLap nThread elapsedTimeFromBnpy elapsedTimeFromProf" \
      > "$BASEDIR/$AllocModel-results-header.txt"

done
