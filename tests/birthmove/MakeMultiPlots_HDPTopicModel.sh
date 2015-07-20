nDocTotal=1024

for nWordsPerDoc in 64 128 256 512
do

for nDocPerBatch in 4 16 64 256
do

python SinglePassBirth_HDPTopicModel.py \
  --nWordsPerDoc $nWordsPerDoc \
  --nDocPerBatch $nDocPerBatch \
  --nDocTotal $nDocTotal \
  --doPause 0

done
done
