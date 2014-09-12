for docID in junkDocs:
  

def plotJunkInDoc(docID, m, Data, B=1.65):
  curLP = m.calc_local_params(Data, nCoordAscentItersLP=5000, convThrLP=.00001)
  curSS = m.get_global_suff_stats(Data, curLP, doPrecompEntropy=1)
  curELBO = m.calc_evidence(SS=curSS)

  docData = Data.select_subset_by_mask([docID])
  docLP = m.calc_local_params(docData, nCoordAscentItersLP=5000, convThrLP=.00001)
  junkids = np.flatnonzero( docLP['resp'][:, 5] > 0.01 )
  pylab.plot(docData.X[:,0], docData.X[:,1], 'k.')
  pylab.plot(docData.X[junkids,0], docData.X[junkids,1], 'rx')
  pylab.axis('image')
  pylab.xlim([-B,B])
  pylab.ylim([-B,B])

  ## Redo by zeroing out junk for this doc
  docLP['DocTopicCount'].fill(100)
  docLP['DocTopicCount'][:,5] = 0
  docLP = m.calc_local_params(docData, docLP, methodLP='memo',
                              nCoordAscentItersLP=5000, convThrLP=.00001)

  print docLP['DocTopicCount']

  newLP = copy.deepcopy(curLP)
  newLP['DocTopicCount'][docID,:] = docLP['DocTopicCount'][0]
  newLP['resp'][Data.doc_range[docID]:Data.doc_range[docID+1]] = docLP['resp']
  newSS = m.get_global_suff_stats(Data, newLP, doPrecompEntropy=1)
  newELBO = m.calc_evidence(SS=newSS)

  print curSS.N
  print newSS.N

  print curELBO
  print newELBO
  pylab.xlabel( 'with junk=%.5f  without=%.5f' \
                 % (curELBO, newELBO))
  if newELBO > curELBO:
    pylab.title('isBetter? YES')
  else:
    pylab.title('isBetter? NO')
