# Initialize build folder
cd ~/llama.cpp
rm -rf build
mkdir -p build
cd build
export CUDACXX=/usr/local/cuda/bin/nvcc

# GPU-only build
cmake .. \
 -DGGML_CUDA=ON \
 -DCMAKE_CUDA_ARCHITECTURES=87 \
 -DLLAMA_BUILD_TESTS=OFF \
 -DLLAMA_BUILD_EXAMPLES=OFF
make -j$(nproc)

# Run main server
chmod +x scripts/start_all_gpu.sh
./scripts/start_all_gpu.sh