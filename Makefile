# Compiler
NVCC = nvcc

# Target
TARGET = app

# Source files
SRC = src/main.cu

# Flags
CFLAGS = -std=c++17 `pkg-config --cflags --libs opencv4`

# Default rule
all:
	$(NVCC) $(SRC) -o $(TARGET) $(CFLAGS)

# Run example (optional helper)
run:
	./app data/input/textures data/output --mode=edge

# Clean build
clean:
	rm -f $(TARGET)