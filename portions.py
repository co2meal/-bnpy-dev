import time
import bnpy


for i in range(1,13):
	for k in range(1,5):
		f=open('overalltimes'+str(k)+'.txt','a') 
		start=time.time()
		bnpy.run('BarsK10V900','FiniteTopicModel','Mult','pVB',nDocTotal=15000,nWordsPerDoc=200,K=50,nLap=1,nCoordAscentItersLP=100,convThrLP=.000001,initname='randomlikewang',lam=.1,alpha=.5,nWorkers=i)
		duration=time.time()-start
		f.write("Overall time was"+str(duration)+'\n')


#NOTE: overalltimes and overallportiontimes are with 1 lap only!!
