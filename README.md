# AIcrowd-AIBlitz8-Solution

## Challenge1 F1 TEAM CLASSIFICATION<br>
Put all the data into the data/ directory and run all cells of run_all.ipynb.<br>
**data/train/.jpg, data/val/.jpg, data/test/.jpg, data/train.csv, data/val.csv, data/sample_submission.csv**

## Challenge2 F1 CAR DETECTION<br>
Put all the data into the data/ directory and run all cells of run.ipynb.<br>
Validation data is not needed here.<br>
**data/train/.jpg, data/test/.jpg, data/train.csv, data/sample_submission.csv**

## Challenge3 F1 SPEED RECOGNITION<br>
I have used code from https://github.com/clovaai/deep-text-recognition-benchmark<br>
Download TPS-ResNet-BiLSTM-CTC.pth from https://drive.google.com/drive/folders/15WPsuPJDCzhp2SvYZLRj8mAlT3zmoAMW and put it into challenge3/TPS-ResNet-BiLSTM-CTC.pth<br>
The link is from the README of the repository which I have used for my code<br>

Put all the data into the data/ directory and run all cells of run.ipynb.<br>
When training is finished write the path of the latest model to the line wich is marked with a comment. The checkpoints are saved in the directory: ligtning_logs/version_/<br>
**data/train/.jpg, data/val/.jpg, data/test/.jpg, data/train.csv, data/val.csv, data/sample_submission.csv**

## Challenge4 F1 CAR ROTATION<br>
> :warning: First train CHALLENGE2 to train this challenge because it needs the model from it.<br>
> If trained challenge2 copy the file **best.pth** into **challenge4/best.pth**<br>
> Put all the data into the data/ directory.<br>
> Run the script preprocess.py. It will cutout the car and save it under the directory **data/preprocessed/train, val, test**<br>
> This will take 4GB of space on disk.
> 
>If you don't want to train challenge2 you can download the weights from: https://drive.google.com/file/d/1UsSfkFvHbMJLOYkdSqXrlO7C_LL2zDGs/view?usp=sharing

Run all cells of run.ipynb.<br>
When training is finished write the path of the latest model to the line wich is marked with a comment. The checkpoints are saved in the directory: ligtning_logs/version_/<br>

## Challenge5 F1 SMOKE ELIMINATION<br>
Just follow the instructions on the README.md in the folder challenge5.
