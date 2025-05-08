EXT_NAME := _C
SRC := als_contractions.cpp cholesky_solve.cpp
OUT := $(patsubst %.cpp,%.so,$(SRC))

# Python and PyTorch include/lib paths
PYTHON_INCLUDE := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
ABIFLAG := -D_GLIBCXX_USE_CXX11_ABI=$(shell python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

TORCH_INCLUDE := $(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.include_paths()))")
TORCH_LIB := $(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.library_paths()))")

INCLUDES := -I$(PYTHON_INCLUDE) $(foreach p,$(TORCH_INCLUDE),-I$(p))
LIBS := $(foreach p,$(TORCH_LIB),-L$(p)) -ltorch -ltorch_cpu -lc10

# Backend-specific linker flags
CUTENSOR_LFLAGS := -L/usr/lib/x86_64-linux-gnu -lcutensor
CUSOLVER_LFLAGS := -L/usr/lib/x86_64-linux-gnu -lcusolver -lcudart

CXX := g++
CXXFLAGS := -O3 -fPIC -std=c++17 $(ABIFLAG)

all: $(OUT)

als_contractions.so: als_contractions.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -shared $(LIBS) $(CUTENSOR_LFLAGS)

cholesky_solve.so: cholesky_solve.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -shared $(LIBS) $(CUSOLVER_LFLAGS)

clean:
	rm -f $(OUT)

.PHONY: all clean
