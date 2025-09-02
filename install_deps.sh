conda create -n mtpi python=3.10 -y
conda activate mtpi
pip install -r requirements.txt

# Third-party dependencies
mkdir mt_pi/third_party
cd mt_pi/third_party

# HaMeR
git clone --recursive https://github.com/geopavlakos/hamer.git
cd hamer
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
pip install -e .[all]
pip install -v -e third-party/ViTPose
cd ..

# SAM 2
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .
cd ..

# Depth Anything
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
pip install -r requirements.txt
cd .. 

conda install -c conda-forge pyrealsense2 -y
pip install open3d

export PYTHONPATH=${PWD}:$PWD/third_party/Depth-Anything-V2