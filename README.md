Skip to content
Search or jump to…
Pull requests
Issues
Marketplace
Explore
 
@giaabaoo 
giaabaoo
/
Mediaeval21-Baseline
Public
forked from nttung1110/Mediaeval21-Baseline
0
0
1
Code
Pull requests
Actions
Projects
Wiki
Security
Insights
Settings
Mediaeval21-Baseline
/
README.md
in
master
 

Spaces

4

Soft wrap
1
# Baseline-Mediaeval21-SportClassificationTask
2
​
3
## Introduction
4
This is the implementation for the baseline of our methods in Mediaeval 2021 Challenge.
5
​
6
# HCMUS at Mediaeval21-SportClassificationTask
7
## Introduction
8
This is the official repository for our baseline [Mediaeval Challenge-Sport Classification Task](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/).
9
Our solution for the problem consists of two stage: individual classification for each component of raw labels, and conditional probability model for producing final results.
10
Here is the general pipeline of our baseline.
11
​
12
## Repository Usage
13
​
14
The original mmpose modules can be found at [Open-mmlab mmpose](https://github.com/open-mmlab/mmpose)
15
​
16
### Data
17
​
18
Since the data for Mediaeval21 SportClassification Task was private, readers are suggested to contact [Challenge owners](https://multimediaeval.github.io/editions/2021/tasks/sportsvideo/)
19
for downloading the data. After that, a sequence of data processing steps was applied before input to our method.
20
​
21
​
22
### Training
23
Run the shell script ```train.sh``` to train
24
​
25
#### List of Arguments accepted
26
```--lr``` Learning rate (Default = 0.002) <br>
27
```--batch_size``` Batch size (Default = 8) <br>
28
```--weight_decay``` Weight decay rate (Default = 1e-4) <br>
29
```--num_workers``` Number of workers (Default = 4) <br>
30
```--epochs``` Number of epochs (Default = 200) <br>
31
```--optimizer``` Optimizer (Default = adagrad) <br>
32
```--frame_interval``` The frame interval (Default = 30) <br>
33
```--img_size``` Size of image (Default = [30, 120, 120]) <br>
34
```--img_hand_size``` Size of the hands (Default = [30, 120, 240]) <br>
35
```--path_pretrained_model``` Path for pretrained model (Default = None) <br>
36
```--human_box_only``` Use human bounding boxes only (Default = True) <br>
37
```--refined_kp``` Get the main player's key points. Should be set to True (Default = False) <br>
38
​
39
​
40
### Evaluating
41
For each classifier, list of probability scores were produced. We utilized all three lists with the help of our proposed conditional probability models with prior knowledge
42
to combine them together as final result. For more information about the methods, please refer to our [Working Notes]().
43
​
44
​
No file chosen
Attach files by dragging & dropping, selecting or pasting them.
@giaabaoo
Commit changes
Commit summary
Create README.md
Optional extended description
Add an optional extended description…
 Commit directly to the master branch.
 Create a new branch for this commit and start a pull request. Learn more about pull requests.
 
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
1
