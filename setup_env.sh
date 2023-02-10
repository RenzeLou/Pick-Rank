# create conda env and setup the main requirements
echo "Creating conda env and installing requirements"
conda create -n pick_rank python==3.8.0
conda activate pick_rank
pip install -r requirements.txt

echo "Installing pytorch-1.12.1+cu113 from wheel"
# install pytorch
# you can find the version you need at https://download.pytorch.org/whl/torch_stable.html
wget https://download.pytorch.org/whl/cu113/torch-1.12.1%2Bcu113-cp38-cp38-linux_x86_64.whl
wget https://download.pytorch.org/whl/cu113/torchvision-0.13.1%2Bcu113-cp38-cp38-linux_x86_64.whl
pip install torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.13.1+cu113-cp38-cp38-linux_x86_64.whl
rm torch-1.12.1+cu113-cp38-cp38-linux_x86_64.whl
rm torchvision-0.13.1+cu113-cp38-cp38-linux_x86_64.whl

echo "Installing spacy"
# install spacy
python -m spacy download en_core_web_sm