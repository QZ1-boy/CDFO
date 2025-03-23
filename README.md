# CDFO

The code of the paper "Deep Compressed Video Super-Resolution With Guidance of Coding Priors". 

# Requirements

CUDA==11.6 Python==3.7 Pytorch==1.13

## 1.1 Environment
```python
conda create -n CDFO python=3.7 -y && conda activate CDFO

git clone --depth=1 https://github.com/QZ1-boy/CDFO && cd QZ1-boy/CDFO/

# given CUDA 11.6
python -m pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116

python -m pip install tqdm lmdb pyyaml opencv-python scikit-image
```

## 1.2 CVCP training dataset and Test dataset
**Download raw HR videos and compressed LR videos**  [CD-VSR]([https://ieeexplore.ieee.org/document/6317156](https://ieeexplore.ieee.org/abstract/document/9509352))


# Train
```python
python train_CPGA.py --opt_path ./config/option_CPGA_vcp_LDB_22.yml
```
# Test
```python
python test_CPGA.py --opt_path ./config/option_CPGA_vcp_LDB_22.yml
```
# Citation
If this repository is helpful to your research, please cite our paper:
```python
@article{zhu2024deep,
  title={Deep compressed video super-resolution with guidance of coding priors},
  author={Zhu, Qiang and Chen, Feiyu and Liu, Yu and Zhu, Shuyuan and Zeng, Bing},
  journal={IEEE Transactions on Broadcasting},
  year={2024},
  publisher={IEEE}
}
```
