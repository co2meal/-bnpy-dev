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
    
    def get_collection_data(self, n):
        '''
        Get the nth tree in the array
        '''
        pass

    def get_parent_index(self, n):
        '''
        Get the index of nth node's parent
        '''
        pass
    
    