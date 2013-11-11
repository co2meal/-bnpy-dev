'''
Unit-tests for ParamBag
'''

from bnpy.suffstats.ParamBag import ParamBag
import numpy as np
import unittest

class TestParamBag(unittest.TestCase):
  def shortDescription(self):
    return None
  
  ######################################################### Verify remove
  def test_removeComp_K1_D1(self):
    A = ParamBag(K=1,D=1)
    A.setField('N', 1, dims='K')
    A.setField('x', 1, dims=('K','D'))
    A.removeComp(0)
    assert A.K == 0
  
  def test_removeComp_K3_D1(self):
    A = ParamBag(K=3,D=1)
    A.setField('N', [1,2,3], dims='K')
    A.setField('x', [4,5,6], dims=('K','D'))
    Aorig = A.copy()
    A.removeComp(1)
    assert Aorig.K == A.K + 1
    assert A.N[0] == Aorig.N[0]
    assert A.N[1] == Aorig.N[2]
    assert np.allclose( A.x, [4,6])

  ######################################################### Verify get
  def test_getComp_K1_D1(self):
    A = ParamBag(K=1,D=1)
    A.setField('N', 1, dims='K')
    A.setField('x', 1, dims=('K','D'))
    c = A.getComp(0)
    assert c.K == 1
    assert c.N == A.N
    assert c.x == A.x
    assert id(c.N) != id(A.N)

  def test_getComp_K3_D1(self):
    A = ParamBag(K=3,D=1)
    A.setField('N', [1,2,3], dims='K')
    A.setField('x', [4,5,6], dims=('K','D'))
    c = A.getComp(0)
    assert c.K == 1
    assert c.N == A.N[0]
    assert c.x == A.x[0]

  ######################################################### Verify add/subtract
  def test_add_K1_D1(self):
    A = ParamBag(K=1,D=1)
    B = ParamBag(K=1,D=1)
    C = A + B
    assert C.K == A.K and C.D == A.D
    A.setField('N', 1, dims='K')
    B.setField('N', 10, dims='K')
    C = A + B
    assert C.N == 11.0

  def test_add_K3_D2(self, K=3, D=2):
    A = ParamBag(K=K,D=D)
    A.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    B = ParamBag(K=K,D=D)
    B.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    C = A + B
    assert np.allclose(C.xxT, A.xxT + B.xxT)

  def test_sub_K3_D2(self, K=3, D=2):
    A = ParamBag(K=K,D=D)
    A.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    B = ParamBag(K=K,D=D)
    B.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    C = A - B
    assert np.allclose(C.xxT, A.xxT - B.xxT)

  def test_iadd_K3_D2(self, K=3, D=2):
    A = ParamBag(K=K,D=D)
    A.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    A.setField('x', np.random.randn(K,D), dims=('K','D'))
    B = ParamBag(K=K,D=D)
    B.setField('x', np.random.randn(K,D), dims=('K','D'))
    B.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    origID = hex(id(A))
    A += B
    newID = hex(id(A))
    assert origID == newID
    A = A + B
    newnewID = hex(id(A))
    assert newnewID != origID

  def test_isub_K3_D2(self, K=3, D=2):
    A = ParamBag(K=K,D=D)
    A.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    A.setField('x', np.random.randn(K,D), dims=('K','D'))
    B = ParamBag(K=K,D=D)
    B.setField('x', np.random.randn(K,D), dims=('K','D'))
    B.setField('xxT', np.random.randn(K,D,D), dims=('K','D','D'))
    origID = hex(id(A))
    A -= B
    newID = hex(id(A))
    assert origID == newID
    A = A - B
    newnewID = hex(id(A))
    assert newnewID != origID
    
  ######################################################### Dim 0 parsing
  def test_parseArr_dim0_passes(self):
    ''' Verify can parse arrays with 0-dim
    '''
    PB1 = ParamBag(K=1, D=1)
    x = PB1.parseArr(1.23, dims=None)
    assert x.ndim == 0 and x.size == 1
    x = PB1.parseArr(1.23, dims=('K'))
    assert x.ndim == 0 and x.size == 1

    PB2 = ParamBag(K=2, D=1)
    x = PB2.parseArr(1.23, dims=None)
    assert x.ndim == 0 and x.size == 1

    PB5 = ParamBag(K=5, D=40)
    x = PB5.parseArr(1.23, dims=None)
    assert x.ndim == 0 and x.size == 1

  def test_parseArr_dim0_fails(self):
    ''' Verify fails for 0-dim input when K > 1
    '''
    PB2 = ParamBag(K=2, D=1)
    with self.assertRaises(ValueError):
      x = PB2.parseArr(1.23, dims=('K'))
    with self.assertRaises(ValueError):
      x = PB2.parseArr(1.23, dims='K')
    

  ######################################################### Dim 1 parsing
  def test_parseArr_dim1_passes(self):
    # K = 1, D = 1
    PB1 = ParamBag(K=1, D=1)
    x = PB1.parseArr([1.23], dims='K')
    assert x.ndim == 0 and x.size == 1
    x = PB1.parseArr([1.23], dims=('K','D'))
    assert x.ndim == 0 and x.size == 1

    # K = *, D = 1
    PB2 = ParamBag(K=2, D=1)
    x = PB2.parseArr([1.,2.], dims='K')
    assert x.ndim == 1 and x.size == 2
    x = PB2.parseArr([1.,2.], dims=('K','D'))
    assert x.ndim == 1 and x.size == 2

    # K = 1, D = *
    PB3 = ParamBag(K=1, D=3)
    x = PB3.parseArr([1., 2., 3.], dims=('K','D'))
    assert x.ndim == 1 and x.size == 3

    # K = *, D = *
    PB2 = ParamBag(K=4, D=1)
    x = PB2.parseArr([1.,2.,3.,4.], dims='K')
    assert x.ndim == 1 and x.size == 4
    x = PB2.parseArr([1.,2.,3.,4.], dims=('K','D'))
    assert x.ndim == 1 and x.size == 4

  def test_parseArr_dim1_fails(self):
    PB2 = ParamBag(K=2, D=1)
    with self.assertRaises(ValueError):
      x = PB2.parseArr([1.23], dims=('K'))
    with self.assertRaises(ValueError):
      x = PB2.parseArr([1.23], dims=('K','D'))

    PB3 = ParamBag(K=1, D=3)
    with self.assertRaises(ValueError):
      x = PB3.parseArr([1.,2.], dims=('K','D'))

    PB3 = ParamBag(K=2, D=3)
    with self.assertRaises(ValueError):
      x = PB3.parseArr([1.,2.,3.,4.,5.,6.], dims=('K','D'))


  ######################################################### Dim 2 parsing
  def test_parseArr_dim2_passes(self):
    PB2 = ParamBag(K=2, D=2)
    x = PB2.parseArr(np.eye(2), dims=('K','D'))
    assert x.ndim == 2 and x.size == 4

    PB1 = ParamBag(K=1, D=2)
    x = PB1.parseArr(np.eye(2), dims=('K','D','D'))
    assert x.ndim == 2 and x.size == 4

    PB31 = ParamBag(K=3, D=1)
    x = PB31.parseArr([[10,11,12]], dims=('K','D'))
    assert x.ndim == 1 and x.size == 3

  def test_parseArr_dim2_fails(self):
    PB2 = ParamBag(K=2, D=2)    
    with self.assertRaises(ValueError):
      x = PB2.parseArr([[1.,2]], dims=('K'))

    with self.assertRaises(ValueError):
      x = PB2.parseArr([[1.,2]], dims=('K','D'))

    with self.assertRaises(ValueError):
      x = PB2.parseArr(np.eye(3), dims=('K','D'))


  ######################################################### Dim 3 parsing
  def test_parseArr_dim3_passes(self):
    K=2
    D=2
    PB = ParamBag(K=K, D=D)
    x = PB.parseArr(np.random.randn(K,D,D), dims=('K','D','D'))
    assert x.ndim == 3 and x.size == K*D*D

    K=1
    D=2
    PB = ParamBag(K=K, D=D)
    x = PB.parseArr(np.random.rand(K,D,D), dims=('K','D', 'D'))
    assert x.ndim == 2 and x.size == K*D*D

    K=3
    D=1
    PB = ParamBag(K=K, D=D)
    x = PB.parseArr(np.random.rand(K,D,D), dims=('K','D', 'D'))
    assert x.ndim == 1 and x.size == K*D*D

  def test_parseArr_dim3_fails(self):
    PB = ParamBag(K=2, D=2)    
    with self.assertRaises(ValueError):
      x = PB.parseArr([[[1.,2]]], dims=('K'))

    with self.assertRaises(ValueError):
      x = PB.parseArr([[[1.,2]]], dims=('K','D'))

    with self.assertRaises(ValueError):
      x = PB.parseArr(np.random.randn(3,3,3), dims=('K','D'))

    with self.assertRaises(ValueError):
      x = PB.parseArr(np.random.randn(3,3,3), dims=('K','D','D'))

