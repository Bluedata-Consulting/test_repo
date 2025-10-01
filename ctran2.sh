set -e

CMAKE_CMD=$(which cmake)

git clone --recursive https://github.com/OpenNMT/CTranslate2.git
cd CTranslate2

mkdir -p build && cd build

$CMAKE_CMD .. \
  -DWITH_CUDA=ON \
  -DWITH_CUDNN=ON \
  -DWITH_MKL=OFF \
  -DOPENMP_RUNTIME=NONE

make -j$(nproc)
sudo make install
sudo ldconfig

# 5. Build the Python bindings
cd ../python
uv pip install -r install_requirements.txt
python setup.py bdist_wheel
uv pip install dist/*.whl

echo "CTranslate2 with CUDA support built and installed successfully!"
