ARCH := sm_75

# Compiler 
NVCC := nvcc
# NVCC_FLAGS := -arch=$(ARCH) -Wno-deprecated-gpu-targets -O2
NVCC_FLAGS := -arch=$(ARCH) \
			  -O2 \
			  -Xcompiler\
			  -fsanitize=address \
			  -g \
			  -Wno-deprecated-gpu-targets \

# Targets
BUILD_DIR := target
TARGET := $(BUILD_DIR)/main
SRC := main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET)

memcheck: $(TARGET)
	compute-sanitizer --tool memcheck ./$(TARGET)

racecheck: $(TARGET)
	compute-sanitizer --tool racecheck ./$(TARGET)

clean:
	rm -f ./target/$(TARGET)
