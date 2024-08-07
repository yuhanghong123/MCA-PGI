# MCA-PGI: Nonlinear Multi-Head Cross-Attention Network and Programmable Gradient Information for Gaze Estimation
The Pytorch Implementation of “MCA-PGI: Nonlinear Multi-Head Cross-Attention Network and Programmable Gradient Information for Gaze Estimation”.

## Requirements

We bulid the project with python=3.8.

```python
pip install -r requirements.txt
```

## Dataset

The datasets used in this study include [ETH-XGaze](https://ait.ethz.ch/xgaze) (pre-trained), [MPIIFaceGaze](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/its-written-all-over-your-face-full-face-appearance-based-gaze-estimation), [EyeDiap](https://www.idiap.ch/en/scientific-research/data/eyediap), and [Gaze360](https://gaze360.csail.mit.edu/)).

If you use these datasets, please cite:

```latex
@inproceedings{Zhang_2020_ECCV,
    author    = {Xucong Zhang and Seonwook Park and Thabo Beeler and Derek Bradley and Siyu Tang and Otmar Hilliges},
    title     = {ETH-XGaze: A Large Scale Dataset for Gaze Estimation under Extreme Head Pose and Gaze Variation},
    year      = {2020},
    booktitle = {The European Conference on Computer Vision (ECCV)}
}
```

```latex
@inproceedings{zhang2017s,
    title={It’s written all over your face: Full-face appearance-based gaze estimation},
    author={Zhang, Xucong and Sugano, Yusuke and Fritz, Mario and Bulling, Andreas},
    booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
    pages={2299--2308},
    year={2017},
    organization={IEEE}
}
```

```latex
@inproceedings{eyediap,
    author = {Funes Mora, Kenneth Alberto and Monay, Florent and Odobez, Jean-Marc},
    title = {EYEDIAP: A Database for the Development and Evaluation of Gaze Estimation Algorithms from RGB and RGB-D Cameras},
    year = {2014},
    isbn = {9781450327510},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/2578153.2578190},
    doi = {10.1145/2578153.2578190},
    booktitle = {Proceedings of the Symposium on Eye Tracking Research and Applications},
    pages = {255–258},
    numpages = {4},
    keywords = {natural-light, database, RGB-D, RGB, remote sensing, gaze estimation, depth, head pose},
    location = {Safety Harbor, Florida},
    series = {ETRA '14}
}
```

```latex
@InProceedings{Kellnhofer_2019_ICCV,
    author = {Kellnhofer, Petr and Recasens, Adria and Stent, Simon and Matusik, Wojciech and Torralba, Antonio},
    title = {Gaze360: Physically Unconstrained Gaze Estimation in the Wild},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
```

## Data Preprocessing

This project utilizes data processing codes provided in [GazeHub](http://phi-ai.org/GazeHub/). You can direct run the method' code using the processed dataset. 

## Usage

**Directly use our code.**

We have implemented the model, with the specific code and training framework referenced from [yihuacheng](https://github.com/yihuacheng/Gaze360.git). 

## Citation

Part of the code implementation draws from [YOLOv9](https://github.com/WongKinYiu/yolov9.git) and [EMTCAL](https://github.com/TangXu-Group/Remote-Sensing-Images-Classification.git).











