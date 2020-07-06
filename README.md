## Dependencies
tested with Ubuntu (18.04), please install PyTorch first following the official instruction. 

- Python 3.7 
- PyTorch (version = 1.40)
- torchvision
- CUDA-10.0 
- numpy
- scipy
- scikit-image 

## Datasets
decompress all files and put them in data/datasets/

copy dir diligent_test to data/datasets/DiLiGenT/pmsData_crop/

generate cycles dataset:
put cycles_train.m in the dir which have PRPS and PRPS_Diffuse(Spline-Net Dataset) and run cycles_train.m in matlab

## Testing
### Test SDPS-Net on the DiLiGenT main dataset
```shell
# Prepare the DiLiGenT main dataset

# Test SDPS-Net on DiLiGenT main dataset using 10 image
CUDA_VISIBLE_DEVICES=0 python eval/run_stage2.py --test_set 1 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/checkp_20.pth.tar --retrain_s2 data/logdir/UPS_Synth_Dataset/TPAMI/10_images/checkp_20.pth.tar
# Please check the outputs in data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/

# or scripts which tests several sets
sh test.sh
```

## Training
master branch: Dataset of SDPS-Net
PRPS branch: Dataset of Spline-Net
### First stage: train Lighting Calibration Network (LCNet)
```shell
# Train LCNet on synthetic datasets using 10 input images
CUDA_VISIBLE_DEVICES=0 python main_stage1.py --in_img_num 10

# training outputs can be found in data/logdir/UPS_Synth_Dataset/CVPR2019/
# trained model can be found in data/logdir/UPS_Synth_Dataset/CVPR2019/10_images

```
### Second stage: train Normal Estimation Network (NENet)
master branch
```shell
# Train NENet on synthetic datasets using 10 input images
CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 10 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/checkp_20.pth.tar

# Train NENet on synthetic datasets using 10 input images and weights pretrained on Dataset of Spline-Net
CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 10 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/checkp_20.pth.tar --retrain_s2 data/logdir/UPS_PRPS_Dataset/TPAMI/10_images_s2/checkp_10.pth.tar

# training outputs can be found in data/logdir/UPS_Synth_Dataset/TPAMI/
# trained model can be found in data/logdir/UPS_Synth_Dataset/TPAMI/10_images
```
PRPS branch
```shell
# Train NENet on synthetic datasets using 10 input images
CUDA_VISIBLE_DEVICES=0 python main_stage2.py --in_img_num 10 --retrain data/logdir/UPS_Synth_Dataset/CVPR2019/10_images/checkp_20.pth.tar

# training outputs can be found in data/logdir/UPS_Synth_Dataset/TPAMI/
# trained model can be found in data/logdir/UPS_Synth_Dataset/TPAMI/10_images
```
