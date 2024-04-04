Please follow this pattern to plug in a new dataset:
```shell
$NAME:
  - ${DS_TYPE}.jsonl    # labels and paths of images
  - ${DS_TYPE}_dataset  # images
  - classnames.txt
  - info.yaml           # metadata of the dataset
```
If the dataset is shared among different $DS_TYPEs, it is recommended to use soft/symbolic links.