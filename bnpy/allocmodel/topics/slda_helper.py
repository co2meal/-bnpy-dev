import numpy as np

def expand_resp(Data,resp):
	# Expand compact representation of resp matrix into full
	# N_d * nDoc x K matrix
	
	nDoc = Data.nDoc
	
	for d in xrange(nDoc):
		start = Data.doc_range[d]
		stop = Data.doc_range[d+1]
		resp_d = resp[start:stop,:]
		wid_d = Data.word_id[start:stop]
		wc_d = Data.word_count[start:stop]
		
		resp_d_expanded = expand_resp_single_doc(resp_d,wid_d,wc_d)
		
		if d == 0:
			resp_expanded = resp_d_expanded
		else:
			resp_expanded = np.concatenate((resp_expanded,resp_d_expanded))
	
	return resp_expanded
	
def expand_resp_single_doc(resp_d,wid_d,wc_d):
	
	N_d = int(sum(wc_d))
	K = resp_d.shape[1]
	
	resp_d_expanded = np.zeros((N_d,K))
	
	
	tokens = []
	for i in range(len(wid_d)):
		tokens.extend( [wid_d[i] for j in range(int(wc_d[i]))] )
	
			
	for n in range(N_d):
		idx = np.where(wid_d == tokens[n])[0][0]
		resp_d_expanded[n,:] = resp_d[idx,:]	
		
	return resp_d_expanded
