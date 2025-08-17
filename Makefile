ARCH := sm_75

# Compiler 
NVCC := nvcc
NVCC_FLAGS := -arch=$(ARCH) -Wno-deprecated-gpu-targets -O2
# NVCC_FLAGS := -gencode arch=compute_75,code=sm_75 \
#               -gencode arch=compute_75,code=compute_75 \
#               -O2 -Wno-deprecated-gpu-targets


# Targets
BUILD_DIR := target
TARGET := $(BUILD_DIR)/vector_add
SRC := vector_add.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f ./target/$(TARGET)
