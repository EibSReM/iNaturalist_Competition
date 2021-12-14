# iNaturalist_Competition

Script to classify images of plants and animals with the image-based species recognition models created in the context of the paper by [Van Horn et al. 2021](http://arxiv-export-lb.library.cornell.edu/pdf/2103.16483), benchmarking different kinds of models created on basis of the dataset of the [iNaturalist 2021 Comptetition](https://github.com/visipedia/inat_comp/tree/master/2021). 

## How-To

### Setup

1. (recommended but not absolutely necessary) create and activate own (conda) environment 
2. install packages 
```
pip install torch torchvision torchaudio
pip install json
```
3. Clone this repository `git clone https://github.com/EibSReM/iNaturalist_Competition.git` and change to respective directory
4. Download pretrained models from the paper [here](https://cornell.box.com/s/bnyhq5lwobu6fgjrub44zle0pyjijbmw), mentioned in the [papers repository](https://github.com/visipedia/newt/tree/main/benchmark).
5. Adapt path to model and images (folder) in the script
6. Run script 
```
python inference.py
```
7. Find results in `Output.txt`


## References
Van Horn G, Cole E, Beery S, Wilber K, Belongie S, Mac Aodha O, et al. (2021) Benchmarking Representation Learning for Natural World Image Collections. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition 12884‑12893. http://arxiv-export-lb.library.cornell.edu/pdf/2103.16483
