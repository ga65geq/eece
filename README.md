# Early-Exit with Class Exclusion for Efficient Inference of Neural Networks

## Train Model
```angular2html
python --cf ./configs/{alexnet, resnet, vgg}/train/{model_name}_{dataset_name}_BC.yaml
```
BC for our method, CR for the traditional method.

## $\beta$  Search

```angular2html
python --cf ./configs/{alexnet, resnet, vgg}/search/{model_name}_{dataset_name}_BC.yaml
```

## Test

```angular2html
python --cf ./configs/{alexnet, resnet, vgg}/test/{model_name}_{dataset_name}_BC.yaml
```