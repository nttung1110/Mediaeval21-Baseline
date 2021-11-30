# Baseline-Mediaeval21-SportClassificationTask

## Introduction
This is the implementation for the baseline of our methods in Mediaeval 2021 Challenge.

# HCMUS at Mediaeval21-SportClassificationTask
## Introduction
This is the official repository for our baseline [Mediaeval Challenge-Sport Classification Task](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/).
Our solution for the problem consists of two stage: individual classification for each component of raw labels, and conditional probability model for producing final results.
Here is the general pipeline of our baseline.

## Repository Usage

The original mmpose modules can be found at [Open-mmlab mmpose](https://github.com/open-mmlab/mmpose)

### Data

Since the data for Mediaeval21 SportClassification Task was private, readers are suggested to contact [Challenge owners](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/)
for downloading the data. After that, a sequence of data processing steps was applied before input to our method.


### Training
Run the shell script ```train.sh``` to train

#### List of Arguments accepted
```--lr``` Learning rate (Default = 0.002) <br>
```--batch_size``` Batch size (Default = 8) <br>
```--weight_decay``` Weight decay rate (Default = 1e-4) <br>
```--num_workers``` Number of workers (Default = 4) <br>
```--epochs``` Number of epochs (Default = 200) <br>
```--optimizer``` Optimizer (Default = adagrad) <br>
```--frame_interval``` The frame interval (Default = 30) <br>
```--img_size``` Size of image (Default = [30, 120, 120]) <br>
```--img_hand_size``` Size of the hands (Default = [30, 120, 240]) <br>
```--path_pretrained_model``` Path for pretrained model (Default = None) <br>
```--human_box_only``` Use human bounding boxes only (Default = True) <br>
```--refined_kp``` Get the main player's key points. Should be set to True (Default = False) <br>


### Evaluating
For each classifier, list of probability scores were produced. We utilized all three lists with the help of our proposed conditional probability models with prior knowledge
to combine them together as final result. For more information about the methods, please refer to our [Working Notes]().


### Testing
Run the shell script ```infer.sh``` to conduct inference on the dataset

### Citation
```
@misc{mmpose2020,
    title={OpenMMLab Pose Estimation Toolbox and Benchmark},
    author={MMPose Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpose}},
    year={2020}
}
```

### Contact Information

If you have any concerns about this project, please contact:

+ Nguyen Trong Tung(nguyentrongtung11101999@gmail.com)

