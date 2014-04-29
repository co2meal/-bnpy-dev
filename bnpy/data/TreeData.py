'''
TreeData.py

Abstract class for tree data objects
'''

class TreeData(object):
    
    def get_child_indices(self, n):
        '''
        Given a node, return indices for children
        '''
        pass
    
    def get_single_tree(self, n):
        '''
        Get the nth tree in the array
        '''
        pass

    def get_parent_index(self, n):
        '''
        Get the index of nth node's parent
        '''
        pass

    def set_mask(self, nBranches):
        '''
        Set mask vectors to collect the nodes on the same branch
        '''
        pass
    
    