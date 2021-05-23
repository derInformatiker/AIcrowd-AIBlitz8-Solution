## Instructions for recreating my solution with pix2pixHD

### Copied repository from: https://github.com/NVIDIA/pix2pixHD

Copy **train/smoke/0-20000.jpg** into **data/f1/train_A** and **train/clear/0-20000.jpg** into **data/f1/train_B**

Copy **test/smoke/0-5000.jpg** into **datasets/f1/test_A**


### For training:<br>
python train.py --name f1 --fp16 --label_nc 0 --dataroot data/f1/ --resize_or_crop none --no_instance --niter 100 --niter_decay 0

### For testing:<br>
python test.py --name f1 --fp16 --label_nc 0 --dataroot datasets/f1/ --resize_or_crop none --no_instance --how_many 5000

The results will be in the **results/f1/test_latest/images**

I've used apex for training wich can lead to a different score if not used.
