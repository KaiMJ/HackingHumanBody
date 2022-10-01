# HackingHumanBody

Cell Segmentation of FTU

How to train network:

```
python3 train.py --number 1 --description "Train First Time" --epochs 100 --batch-size 16 --learning-rate 0.001 --optimizer "adam" --weight-decay 0.99 --split 0.9 --img-size 256
```

**SETUP**

Create models and tensorboard dirrectories.

```
mkdir 'saved/{models, tensorboard}'
```

Logs will be saved to logs.txt

Ex.

```
logs
```

**Magnetic Resonance Imaging scans used for pretraining**

Lower-grade glioma tumors are marked as green areas.

FLAIR: Fluid-attenuated inversion recovery reveals tissue 

<a href="https://www.sciencedirect.com/science/article/abs/pii/S0010482519301520">Original Paper</a>


<p float="left">
  <img src="images/TCGA_DU_6408_19860521.gif" width="200" />
  <img src="images/TCGA_DU_7301_19911112.gif" width="200" /> 
  <img src="images/TCGA_FG_5962_20000626.gif" width="200" />
  <img src="images/TCGA_HT_7690_19960312.gif" width="200" />
  <img src="images/TCGA_HT_7693_19950520.gif" width="200" />
  <img src="images/TCGA_HT_7694_19950404.gif" width="200" />
</p>

**RESULTS**

## Focal + Dice Loss

![Alt text](images/1.png?raw=true "Focal + Dice Loss")

## Dice Loss

![Alt text](images/2.png?raw=true "Dice Loss")

## Train Predicted

![Alt text](images/3.png?raw=true "Train Predicted")

## Val Predicted

![Alt text](images/4.png?raw=true "Validation Predicted")
