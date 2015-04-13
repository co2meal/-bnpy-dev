#!/contrib/projects/EnthoughtPython/epd64/bin/python
import time
import bnpy


for i in range(1,13):
		#f=open('wikioveralltimes'+str(k)+'.txt','a') 
		#start=time.time()
	bnpy.run('wiki','FiniteTopicModel','Mult','pVB',nDocTotal=15000,nWordsPerDoc=200,K=50,nLap=1,nCoordAscentItersLP=100,convThrLP=.000001,initname='randomlikewang',lam=.1,alpha=.5,nWorkers=1)	#	duration=time.time()-start		#f.write(str(duration)+'\n')


#NOTE: overalltimes and overallportiontimes are with 1 lap only!!
