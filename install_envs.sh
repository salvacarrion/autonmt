# Create conda env
conda create --name autonmt python=3.8
conda activate autonmt
#conda remove --name mltests --all

# Install autonmt
pip install -e .

# Create fairseq env
conda deactivate autonmt
pip install virtualenv
mkdir -p ~/venvs
virtualenv -p $(which python) ~/venvs/fairseq
source ~/venvs/fairseq/bin/activate

# Install fairseq from source
git clone git@github.com:salvacarrion/fairseq.git
cd fairseq/
pip install -e .

# Install sentencepiece from source
# See: https://github.com/google/sentencepiece
#sudo apt-get install cmake build-essential pkg-config libgoogle-perftools-dev
#git clone https://github.com/google/sentencepiece.git
#cd sentencepiece
#mkdir build
#cd build
#cmake ..
#make -j $(nproc)
#sudo make install
#sudo ldconfig -v