import os
import nose

testroot = os.path.sep.join(os.path.abspath(__file__).split(os.path.sep)[:-1])
if not testroot.endswith(os.path.sep):
    testroot = testroot + os.path.sep

CMD = "%s/TestKMeans.py:TestN1000K10.test_speed -v --nocapture " % (testroot)
result = nose.run(argv=CMD.split())
