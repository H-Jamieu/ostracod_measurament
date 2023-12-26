# The code for measure ostracods
## Steps:
* 1. Run `mask_generation.py` to get the masks. Noted [segment-anything](https://github.com/facebookresearch/segment-anything) should be installed to run the mask generation normally.
* 2. Run `mask_postprocessing.py` to remove the extra non-connected components in the masks.
* 3. Run `Getting_scalebar.py` to get the length of the scale bar in pixel. A file named `scalebar_length.csv` will be created under this directory to record scale-bar to file relationship.
* 4. Run `measure-masks.py` to get measurament. Note this step is super slow!

## Todo
* Update readme
* Configure multi-processing of `measure-masks.py` to increase speed. Should be at least 4X faster depends on user CPU.
* Create on-click run script to increase usability.
* Draw the design diagram for the project
