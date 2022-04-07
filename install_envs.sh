# Create conda env
conda create --name autonmt python=3.8
conda activate autonmt
#conda remove --name mltests --all

# Install autonmt
pip install -e .