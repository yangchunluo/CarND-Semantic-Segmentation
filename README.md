## Semantic Segmentation Project

Yangchun Luo<br>
July 29, 2018

This is the assignment for Udacity's Self-Driving Car Term 3 Project 2.

---

The goal of this project is to label the pixels of a road in images using a Fully Convolutional Network (FCN).

### To run

```bash
python main.py
```

Other setup information can be found in the original [README](README-orig.md) file.

### Train loss

Training loss is decreasing over time, as shown the plot below. (plotted by `plot_training_loss.py` based on [training.log](training.log))

<img src='./training_loss.png' />

### Results

<img src='./runs/1532892790.3617666/um_000000.png' />

<img src='./runs/1532892790.3617666/um_000010.png' />

<img src='./runs/1532892790.3617666/umm_000020.png' />

<img src='./runs/1532892790.3617666/umm_000040.png' />