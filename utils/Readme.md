### discription of each file : 

1. `meanSTD.py` we used this file to calculate mean and STD for each of train/val/test sets. 

note : may need modification for addressing of files 

sample output : 
```
18930/18930 [04:54<00:00, 64.33it/s]
tensor([0.4846, 0.5079, 0.5005])
tensor([0.2687, 0.2705, 0.2869])

—————-
2111/2111 [00:30<00:00, 68.59it/s]
tensor([0.4851, 0.5083, 0.5009])
tensor([0.2690, 0.2709, 0.2877])
```

2. `splt.py` we devided train set to train and validation sets 
`2folders of images train and val sets`
3. `voc2coco.py` : we used this file to covert voc annotation to coco. 
`input folder of xml files and name of jsonfile  output json file with coco style `

4. `createimageinfo.py` this file generate image list with coco style . It was required for some generation of file. 
`input folder of images and name of jsonfile  output json file with coco style(only image names and sizes)`

5. `Pre_Process_hist_equal.ipynb`: the codes in this file make the images' histograms equalized(with some modifications). It is easy to embed this in main code. But we have not yet. So we have included revised images by code in data folder to make the  proccess of checking the model easier.   
`input folder of images output folder of hiseq images`

6. `kmeans_anchors_ratios.py` : A parameter needed for retina net. we used this repo to calculate them. 
`sample output provided inside the code`
