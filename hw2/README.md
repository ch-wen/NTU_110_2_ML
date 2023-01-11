# ML2022Spring Homework 2 

音訊分類

## Preparation
Language: Python 3.9.7  
IDE: Spyder 5.1.5

除了Sample code中的套件外，額外使用了argparse。將training和testing分開避免RAM不足。

```
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode")
    args = parser.parse_args()
    ...
    if args.mode == 'train':
    ...
    if args.mode == 'test':
    ... 
```

## Running the Training and Testing

需在Console中依序執行以下兩行程式碼
```
run r10945006_hw2.py --mode train
run r10945006_hw2.py --mode test
```
train用來產生模型，test用來產生預測結果
## Model
RNN(GRU)+FC
## Data prarameters
concat_nframe設為139  (已達電腦RAM之極限)  
我的想法為concat_nframe越高，準確度應該會越好  
train_ratio設為0.95  
因這次資料量較多，我認為validation 0.05已足夠
```
concat_nframes = 139
train_ratio = 0.95
```
## Result
```
Model: model.ckpt
Prediction: prediction.csv
```
## Authors

* **NTU BEBI 温皆循 R10945006**

