if [[ -z $targetUID ]]; then
    export targetUID=0
fi

if [[ -z $Kinit ]]; then
    export Kinit=2
fi

if [[ -z $nDocTotal ]]; then
    export nDocTotal=1024
fi

outputdir="/tmp/nDoc=$nDocTotal""_Kinit=$Kinit""_targetCompID=$targetUID/"
mkdir -p $outputdir
echo $outputdir

for nWordsPerDoc in 64 128 256 512
do

for nDocPerBatch in 4 16 64 256
do

python SinglePassBirth_HDPTopicModel.py \
  --nWordsPerDoc $nWordsPerDoc \
  --nDocPerBatch $nDocPerBatch \
  --nDocTotal $nDocTotal \
  --Kinit $Kinit \
  --targetUID $targetUID \
  --outputdir $outputdir \
  --doShowAfter 0

done
done
