from scipy.special import digamma, gammaln
import numpy as np


def inferLocal_SingleDoc_Dir(wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                             DocTopicCount_d=None, sumR_d=None,
                             nCoordAscentItersLP=10, convThrLP=0.001, 
                             restartremovejunkLP=0,
                             **kwargs):
  ''' Infer compact local parameters for a single document

      Assumes q(Pi_d) is a Dirichlet.

      Args
      --------

      Kwargs
      --------
      restartremovejunkLP : set to 2 to do doc-level removal of small topics

      Returns
      --------
      DocTopicCount_d : updated doc-topic counts
      Prior_d : prob of topic in document, up to mult. constant
      sumR_d : normalization constant for each token
  ''' 
  if sumR_d is None:
    sumR_d = np.zeros(Lik_d.shape[0])

  ## Initialize using global topic probabilities
  if DocTopicCount_d is None:
    Prior_d = alphaEbeta.copy()
    ## Update sumR_d for all tokens in document
    np.dot(Lik_d, Prior_d, out=sumR_d)

    ## Update DocTopicCounts
    DocTopicCount_d = np.zeros_like(Prior_d)
    np.dot(wc_d / sumR_d, Lik_d, out=DocTopicCount_d)
    DocTopicCount_d *= Prior_d
      
  prevDocTopicCount_d = DocTopicCount_d.copy()
  for iter in xrange(nCoordAscentItersLP):
    ## Update Prob of Active Topics
    np.add(DocTopicCount_d, alphaEbeta, out=Prior_d)
    digamma(Prior_d, out=Prior_d)   # Prior_d = E[ log pi_dk ] + constant
    #Prior_d -= Prior_d.max()
    np.exp(Prior_d, out=Prior_d)    # Prior_d = exp E[ log pi_dk ] / constant
      
    ## Update sumR_d for all tokens in document
    np.dot(Lik_d, Prior_d, out=sumR_d)

    ## Update DocTopicCounts
    np.dot(wc_d / sumR_d, Lik_d, out=DocTopicCount_d)
    DocTopicCount_d *= Prior_d

    ## Check for convergence
    if iter % 5 == 0:
      maxDiff = np.max(np.abs(DocTopicCount_d - prevDocTopicCount_d))
      if maxDiff < convThrLP:
        break
    prevDocTopicCount_d[:] = DocTopicCount_d

  Info = dict(maxDiff=maxDiff, iter=iter)
  if restartremovejunkLP == 2:
    DocTopicCount_d, Prior_d, sumR_d, RInfo = removeJunkTopics_SingleDoc(
                     wc_d, Lik_d, alphaEbeta, alphaEbetaRem, 
                     DocTopicCount_d, Prior_d, sumR_d, **kwargs)
    Info.update(RInfo)
  return DocTopicCount_d, Prior_d, sumR_d, Info


def removeJunkTopics_SingleDoc(wc_d, Lik_d, alphaEbeta, alphaEbetaRem,
                               DocTopicCount_d, Prior_d, sumR_d,
                               restartNumTrialsLP=5,
                               restartNumItersLP=2, 
                               restartMaxThrLP=25,
                               restartCriteriaLP='smallest',
                               MIN_USAGE_THR=0.01, 
                               **kwargs):
  ''' Create candidate models that remove junk topics, accept if improved.
  '''
  Info = dict(nTrial=0, nAccept=0)
  usedTopicMask = DocTopicCount_d > MIN_USAGE_THR
  nUsed = np.sum(usedTopicMask)
  if nUsed < 2:
    return DocTopicCount_d, Prior_d, sumR_d, Info
    
  ## Measure current model quality via ELBO
  curELBO = calcELBO_SingleDoc_Dir(DocTopicCount_d, Prior_d, sumR_d,
                                   wc_d, alphaEbeta, alphaEbetaRem)
  Info['startELBO'] = curELBO

  ## Determine eligible topics to delete
  if restartCriteriaLP == 'DocTopicCount':
    usedTopics = np.flatnonzero(np.logical_and(usedTopicMask,
                                          DocTopicCount_d < restartMaxThrLP))
  else:
    usedTopics = np.flatnonzero(usedTopicMask)
  smallIDs = np.argsort(DocTopicCount_d[usedTopics])[:restartNumTrialsLP]
  smallTopics = usedTopics[smallIDs]
  smallTopics = smallTopics[:nUsed-1]

  pDocTopicCount_d = np.zeros_like(DocTopicCount_d)
  pPrior_d = np.zeros_like(Prior_d)
  psumR_d = np.zeros_like(sumR_d)  

  for kID in smallTopics:
    ## Propose deleting current "small" topic
    pDocTopicCount_d[:] = DocTopicCount_d
    pDocTopicCount_d[kID] = 0
    
    ## Refine initial proposal via standard coord ascent updates
    for iter in xrange(restartNumItersLP):
      # Update Prob of Active Topics
      np.add(pDocTopicCount_d, alphaEbeta, out=pPrior_d)
      digamma(pPrior_d, out=pPrior_d)   # Prior_d = E[ log pi_dk ] + const
      #pPrior_d -= pPrior_d.max()
      np.exp(pPrior_d, out=pPrior_d)    # Prior_d = exp E[ log pi_dk ] / const
     
      # Update sumR_d for all tokens in document
      np.dot(Lik_d, pPrior_d, out=psumR_d)

      # Update DocTopicCounts
      np.dot(wc_d / psumR_d, Lik_d, out=pDocTopicCount_d)
      pDocTopicCount_d *= pPrior_d

    ## Evaluate proposal quality via ELBO
    propELBO = calcELBO_SingleDoc_Dir(pDocTopicCount_d, pPrior_d, psumR_d,
                                      wc_d, alphaEbeta, alphaEbetaRem)
    Info['nTrial'] += 1
    if not np.isfinite(propELBO):
      continue

    ## Update current model if accepted!
    if propELBO > curELBO:
      Info['nAccept'] += 1
      curELBO = propELBO
      DocTopicCount_d[:] = pDocTopicCount_d
      Prior_d[:] = pPrior_d
      sumR_d[:] = psumR_d
      nUsed -= 1

  ## Package up and return
  Info['finalELBO'] = curELBO
  return DocTopicCount_d, Prior_d, sumR_d, Info  


                               
def calcELBO_SingleDoc_Dir(DocTopicCount_d, Prior_d, sumR_d,
                           wc_d, alphaEbeta, alphaEbetaRem):
  ''' Calculate single document contribution to the ELBO (log evidence bound).
  '''
  theta_d = DocTopicCount_d + alphaEbeta

  if alphaEbetaRem is None:
    ## LDA model, with K active topics
    sumTheta = theta_d.sum()
    digammaSum = digamma(sumTheta)
    ElogPi_d = digamma(theta_d) - digammaSum

    L_cDir = np.sum(gammaln(theta_d)) - gammaln(sumTheta)
    # SLACK terms are always equal to zero!
    #L_slack = np.inner(DocTopicCount_d + alphaEbeta - theta_d, ElogPi_d)
  else:
    ## HDP model, with K active topics and one "leftover aggregate mass" topic
    sumTheta = theta_d.sum() + alphaEbetaRem
    digammaSum = digamma(sumTheta)
    ElogPi_d = digamma(theta_d) - digammaSum
    ElogPiRem = digamma(alphaEbetaRem) - digammaSum

    L_cDir = np.sum(gammaln(theta_d)) + gammaln(alphaEbetaRem) \
             - gammaln(sumTheta)
    # SLACK terms are always equal to zero!
    #L_slack = np.inner(DocTopicCount_d + alphaEbeta - theta_d, ElogPi_d) 
    #L_slackRem =  (alphaEbetaRem - thetaRem) * ElogPiRem
            
  L_rest = np.inner(wc_d, np.log(sumR_d)) \
           - np.inner(DocTopicCount_d, np.log(Prior_d))
  return L_cDir + L_rest  
