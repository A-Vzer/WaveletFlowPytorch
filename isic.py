path = 'data/isic'
augmentation = True
affineTransform = True
imageAug = False
in_channels = 3
weight_decay = 5e-5
patch_size = [128,128]
lr = 1e-4

degrees = [-180, 180]
translate = [0.1, 0.1]
scale_ranges = [0.9, 1.1]
shears = [-10, 10, -10, 10]
#
brightness = [0.8, 1.2]
gamma = [0.8, 1.2]