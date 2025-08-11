set -e

git clone --recursive https://github.com/OpenNMT/CTranslate2.git

cd CTranslate2

cmake -Bbuild_folder -DWITH_MKL=OFF -DOPENMP_RUNTIME=NONE -DWITH_CUDA=ON -DWITH_CUDNN=ON

cmake —build build_folder

cd build_folder

sudo make install

cd python

pip install -r install_requirements.txt

python3 setup.py bdist_wheel

pip3 install dist/*.whl