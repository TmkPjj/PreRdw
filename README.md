Our method is based on Depth Anything V2. By finetuning on the Booster training set


## Usage

### Preparation
First, you need to download the pretrained [checkpoint](https://pan.baidu.com/s/1kME4xQJEbvUZUeAgxgQAPg?pwd=6789) to 'weight/'.

Then, update the booster test dataset path in run.py.

### Evaluate

```python
python run.py
python ensemble_light.py 
python round.py 
python filter.py
```


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

