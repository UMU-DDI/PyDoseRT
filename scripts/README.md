Some scripts for experiments

# Work in progress

| ROI                  | lower_bound_gy | higher_bound_gy | lower_bound_target_percent | higher_bound_target_percent | weight |
|----------------------|----------------|-----------------|----------------------------|-----------------------------|--------|
| PTV                  | 74             | 83              | 95                         | 100                         | 10     |
| ROI1 (PenileBulb)    | 0              | 100             | 0                          | 100                         | 1      |
| ROI2 (FemoralHead_L) | 0              | 50              | 0                          | 100                         | 2      |
| ROI3 (FemoralHead_R) | 0              | 50              | 0                          | 100                         | 2      |
| ROI4 (Bladder)       | 0              | 70              | 0                          | 85                          | 2      |
| ROI5 (Rectum)        | 0              | 70              | 0                          | 75                          | 2      |
| ROI6 (Background)    | 0              | 100             | 0                          | 100                         | 1      |


# Experiments
```
CUDA_VISIBLE_DEVICES=1 python scripts/static_optimize_deviation_val_Adam.py --is_comet 0 --is_debug 0 --number_of_cps 3 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 2000 --lr 0.0001

CUDA_VISIBLE_DEVICES=1 python scripts/static_optimize_deviation_val_Adam.py --is_comet 0 --is_debug 0 --number_of_cps 9 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 2000 --lr 0.0001

CUDA_VISIBLE_DEVICES=2 python scripts/static_optimize_deviation_val_Adam.py --is_comet 0 --is_debug 0 --number_of_cps 3 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 2000 --lr 0.0001

CUDA_VISIBLE_DEVICES=2 python scripts/static_optimize_deviation_val_Adam.py --is_comet 0 --is_debug 0 --number_of_cps 9 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 1 --downsampling_factor 2 --epochs 2000 --lr 0.0001



CUDA_VISIBLE_DEVICES=0 nohup python scripts/train.py --is_comet 1 --is_debug 0 --number_of_cps 90 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 0 --downsampling_factor 2 --lr 0.0005 > d0.log &

CUDA_VISIBLE_DEVICES=1 nohup python scripts/train.py --is_comet 1 --is_debug 0 --number_of_cps 60 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 0 --downsampling_factor 2 --lr 0.0005 > d1.log &

CUDA_VISIBLE_DEVICES=2 nohup python scripts/train.py --is_comet 1 --is_debug 0 --number_of_cps 45 --batch_size 1 --initial_filters 16 --vae_n_filters 16 --weight_PTV 10 --constraint_mode fixed --is_load_pretrained 0 --downsampling_factor 2 --lr 0.0005 > d2.log &



```
