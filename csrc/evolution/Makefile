EXT_NAME = _C
SRC = als_ops.cpp
OUT = $(EXT_NAME).so

# CUDA/cuTENSOR installed via APT (system-wide)
CUTENSOR_CFLAGS := -I/usr/include
CUTENSOR_LFLAGS := -L/usr/lib/x86_64-linux-gnu -lcutensor

# Python and PyTorch settings
PYTHON_INCLUDE := $(shell python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])")
ABIFLAG := -D_GLIBCXX_USE_CXX11_ABI=$(shell python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

LIBTORCH_INCLUDE := $(shell python3 -c "from torch.utils import cpp_extension; print(' '.join(cpp_extension.include_paths()))")
LIBTORCH_LIB := $(shell python3 -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])")

TORCH_CFLAGS := $(foreach path,$(LIBTORCH_INCLUDE),-I$(path))
TORCH_LFLAGS := -L$(LIBTORCH_LIB) -ltorch -ltorch_cpu -lc10

CXX = g++
CXXFLAGS = -O3 -fPIC -std=c++17 $(ABIFLAG) -I$(PYTHON_INCLUDE) $(TORCH_CFLAGS) $(CUTENSOR_CFLAGS)
LDFLAGS  = -shared $(TORCH_LFLAGS) $(CUTENSOR_LFLAGS)

all: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f $(OUT)
