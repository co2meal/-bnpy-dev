for canShuffle in 'ByBeta' 'None' 
do
	for canShuffleInit in 'ByUsage' 'ByCounts' # 'Random'
	do
		CMD="
		python TestFromFixedCountsToRhoOmegaBetter.py \
			--dumppath \"/tmp/nips/billings-K=200/1/snapshot.dump\" \
			--maxiter 100 \
			--useSavedInit_omega 0 \
			--useSavedInit_rho 0 \
			--doInteractive 0 \
			--canShuffle $canShuffle \
			--canShuffleInit $canShuffleInit \
			--savename \"initsort=${canShuffleInit}_sort=${canShuffle}\"
		"

		echo $CMD
		eval $CMD
		echo ""

	done
done
