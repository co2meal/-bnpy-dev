all: libfwdbwd

# Rule: compile libfwdbwd from Eigen
# Explanation of flags:
# -O3 : level 3 optimizations make this code blazing fast
# --shared -fPIC : build shared library, which is callable from Python
# -I$EIGENPATH : set the include path so Eigen library can be found
# -w : disable all warnings (makes for cleaner output).
libfwdbwd: hasEigenpath
	cd bnpy/allocmodel/hmm/lib/; \
	g++ FwdBwdRowMajor.cpp -o libfwdbwd.so \
		--shared -DNDEBUG -O3 -I$(EIGENPATH) -fPIC -w;

# Rule: verify that EIGENPATH exists, or instruct user to download it.
hasEigenpath:
ifndef EIGENPATH
		$(error EIGENPATH not set. \
			First, install Eigen (v3+) from eigen.tuxfamily.org. \
			Next, in terminal: export EIGENPATH=/path/to/eigen3/)
endif


