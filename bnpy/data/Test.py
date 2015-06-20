

class Top:
  @classmethod
  def makeme(cls, a):
    return cls(a)

  def __init__(self, a):
    self.a = a



class A(Top):
  def __init__(self, a, b):
    self.a = a
    self.b = b

  def printme(self):
    print 'A AND B = ', self.a, self.b
    
  


top = Top.makeme(5)
print hasattr(top, 'printme')
top.__class__ = A
print hasattr(top, 'printme')
top.b = 44
top.printme()

  
