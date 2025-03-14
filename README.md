# CDFO

The code of paper "Deep Compressed Video Super-Resolution With Guidance of Coding Priors". 

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
## 1.2 DCNv2
```python
cd ops/dcn/
bash build.sh
```
Check if DCNv2 work (optional)
```python
python simple_check.py
```
## 1.3 MFQEv2 dataset
**Download raw and compressed videos** 

**Edit YML**

You need to edit option_TGAF_MFQEv2_#_QP#.yml file.

**Generate LMDB**

The LMDB generation for speeding up IO during training.
```python
python create_vcp.py --opt_path option_CPGA_vcp_#_QP#.yml
```
Finally, the VCP dataset root will be sym-linked to the folder ./data/ automatically.

## 1.4 Test dataset

We use the JCT-VC testing dataset in [JCT-VC](https://ieeexplore.ieee.org/document/6317156). Download raw and compressed videos [BaiduPan](https://pan.baidu.com/s/1IFjZF2MvCyVOmgTBHgl2IA),Code [qix5].

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
@article{zhu2024compressed,
  title={Compressed Video Quality Enhancement with Temporal Group Alignment and Fusion},
  author={Qiang, Zhu and Yajun, Qiu and Yu, Liu and Shuyuan, Zhu and Bing, Zeng},
  journal={IEEE Signal Processing Letters}
}
```
# Related Works
We also released some compressed video quality enhancement models, e.g., [STDF](https://github.com/RyanXingQL/STDF-PyTorch), [RFDA](https://github.com/zhaominyiz/RFDA-PyTorch), [CF-STIF](https://github.com/xiaomingxige/CF-STIF), and  [STDR](https://github.com/xiaomingxige/STDR).
