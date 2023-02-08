## Baselines and VD-DC

Some non-core models can be downloaded from this [link](https://drive.google.com/drive/folders/1SloRnOO9UnonfvubPWfw0uFpLco_2JvH?usp=sharing). 
It contains two baseline models: ```sd-v1-4.pth``` and ```sd-variation.pth```, and our VD-DC model ```vd-dc.pth```

All models should be copyed to ```pretrained``` folder.

To evaluate baseline experiments:

```
python main.py --config sd_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
python main.py --config sd_variation_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
```

You will need to create ```./log/sd_nodataset/99999_eval``` to make these baseline evaluations running.

To evaluate VD-DC experiments:

```
python main.py --config vd_dc_eval --gpu 0 1 2 3 4 5 6 7 --eval 99999
```

Similarly, you will need to create ```./log/vd_nodataset/99999_eval``` to make the evaluation running.
