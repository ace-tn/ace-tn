EXT_NAME = _C
SRC = als_contractions.cpp cholesky_solve.cpp
OUT = $(patsubst %.cpp,%.so,$(SRC))

PYTHON_INCLUDE := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
ABIFLAG := -D_GLIBCXX_USE_CXX11_ABI=$(shell python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

LIBTORCH_INCLUDE := $(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.include_paths()))")
LIBTORCH_LIB := $(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])")

TORCH_CFLAGS := $(foreach path,$(LIBTORCH_INCLUDE),-I$(path))
TORCH_LFLAGS := -L$(LIBTORCH_LIB) -ltorch -ltorch_cpu -lc10

CUTENSOR_FLAGS := -I/usr/include -L/usr/lib/x86_64-linux-gnu -lcutensor
CUSOLVER_FLAGS := -I/usr/include -L/usr/lib/x86_64-linux-gnu -lcusolver -lcudart

CXX = g++
BASE_CXXFLAGS = -O3 -fPIC -std=c++17 $(ABIFLAG) -I$(PYTHON_INCLUDE) $(TORCH_CFLAGS)
BASE_LDFLAGS = -shared $(TORCH_LFLAGS)

all: $(OUT)

als_contractions.so: als_contractions.cpp
	$(CXX) $(BASE_CXXFLAGS) -I/usr/include $< -o $@ $(BASE_LDFLAGS) $(CUTENSOR_FLAGS)

cholesky_solve.so: cholesky_solve.cpp
	$(CXX) $(BASE_CXXFLAGS) -I/usr/include $< -o $@ $(BASE_LDFLAGS) $(CUSOLVER_FLAGS)

clean:
	rm -f $(OUT)
