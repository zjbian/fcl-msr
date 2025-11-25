Our code forked from https://github.com/aHuiWang/CIKM2020-S3Rec

The specific steps are as follows:

## Preparation 
```python
pip install -r requirements.txt
```

## Train/Fintune model
### Note (from https://github.com/aHuiWang/CIKM2020-S3Rec)

When you fine-tune the model, please check the log information. If it is
```
ckp_file Not Found! The Model is same as SASRec.
```
then you actually run the SASRec and the model's parameters are initialized randomly. Otherwise you would see
```
Load Checkpoint From ckp_path!
```
which means you successfully initialize the model with pre-trained parameters.

### Note (from our datasets pretrained models)

we provide our pretraind model on our four datasets in reproduce/(and the ckp is 200 except 150 for Scientific), you can download them first. 

### Train
```python
python run_test.py --output_dir （） --ckp 150 --data_name Scientific --Ours --MMOE
```
## Evaluate
```python
python run_test.py --output_dir （） --ckp 150 --data_name Scientific --Ours --MMOE --do_eval
```

## NOTE

the path of the dataset and the pretrained model should be changed in the code ,and the path of the output_dir should be changed in the run_test.py
```
the  model args can be modified in the code run_test.py
```
If you have any question please leave message at ISSUE.
