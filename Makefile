SHELL := /bin/bash 

all: util_entropy libfwdbwd libsparsetopics libsparseresp

# Rule: compile C extension via cython for fast calculation of entropy
util_entropy:
	cd bnpy/util/; \
	python setup.py build_ext --inplace

# Rule: compile libfwdbwd from Eigen
# Explanation of flags:
# -O3 : level 3 optimizations make this code blazing fast
# --shared -fPIC : build shared library, which is callable from Python
# -I$EIGENPATH : set the include path so Eigen library can be found
# -w : disable all warnings (makes for cleaner output)
# -DNDEBUG : disable all eigen assertions and other runtime checks
libfwdbwd: hasEigenpath
	cd bnpy/allocmodel/hmm/lib/; \
	g++ FwdBwdRowMajor.cpp -o libfwdbwd.so \
		-I$(EIGENPATH) \
		-DNDEBUG -O3 \
		--shared -fPIC -w $(PYARCH);

libsparseresp: hasEigenpath
	echo "$(PYVERSION)"; \
	cd bnpy/util/lib/sparseResp/; \
	g++ SparsifyRespCPPX.cpp -o libsparseresp.so \
		-I$(EIGENPATH) \
		-DNDEBUG -O3 \
		--shared -fPIC -w $(PYARCH);

libsparsetopics: hasEigenpath
	cd bnpy/util/lib/sparseResp/; \
	g++ TopicModelLocalStepCPPX.cpp -o libsparsetopics.so \
		-I$(BOOSTMATHPATH) -I$(EIGENPATH) \
		-DNDEBUG -O3 \
		--shared -fPIC -w $(PYARCH);

# Rule: verify that EIGENPATH exists, or instruct user to download it.
hasEigenpath:
ifndef EIGENPATH
		$(error EIGENPATH not set. \
			First, install Eigen (v3+) from eigen.tuxfamily.org. \
			Next, in terminal: export EIGENPATH=/path/to/eigen3/)
endif

#hasPyVersion:
#ifndef PYVERSION
#	$(export PYVERSION=`python -c "import ctypes; print 8 * ctypes.sizeof(ctypes.c_int)"`)
#	$(export ARCH="-m32"`)
#if [[ $(PYVERSION) -eq '32' ]]; then
#	ARCH="-m32"
#else
#	ARCH=""
#fi
#endif
