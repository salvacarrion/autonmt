# Remove environment
conda remove --name autonmt --all

# Create conda env
conda create --name autonmt python=3.11
conda activate autonmt

# Install AutoNMT
pip install -e .