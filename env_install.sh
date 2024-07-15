conda create -n mind3d python=3.8 -y
conda activate mind3d
pip install numpy cython
python setup.py build_ext --inplace
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
conda install pytorch-scatter -c pyg
pip install -r requirements.txt