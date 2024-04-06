# A Project For The Evalution of GLoRIA

This projects are used for evalute GLoRIA in various datasets.

## Plug in datasets

Refer to [datasets/README.md](datasets/README.md).

We prepared an example dataset(vindr, of a size about 100M) for you [here](https://box.nju.edu.cn/f/3372ee83ce2544b5b5e5/?dl=1) at NJUBOX. You can directly extract the tarball to `./datasets/`.

## Download the pretrained weights

[The original respository of GLoRIA](https://github.com/marshuang80/gloria) provides [the download link](https://github.com/marshuang80/gloria) the pretrained weights of GloRIA on chexpert dataset.

You should put the files in `./pretrained/`.

## Prepare the environment

```shell
conda env create -f environment.yml
conda activate gloria_test
pip install -r requirements.txt -i https://mirror.nju.edu.cn/pypi/web/simple
```

## Execute the evaluation

```shell
make
```

The results for all datasets plugged will be output to `./results/`.