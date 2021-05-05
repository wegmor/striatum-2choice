This repository contains the python code that generated the figures in the publication "Complete representation of action space and value in all striatal pathways", available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.03.29.983825v1). The associated data files can be found here [TODO].

- `endoData_2019.hdf` includes the [CaImAn](https://github.com/flatironinstitute/CaImAn)-extracted calcium traces and behavioral data (operant chamber logs, [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut)-tracked coordinates).
- `alignment_190227.hdf` includes ROI-mappings across days as well as the ROI spatial filters as images.


### How to recreate our figures

Create an isolated python environment and install the required packages using [conda](https://docs.anaconda.com/anaconda/install/).

```bash
$ conda create -n striatum-2choice python=3.8.3
$ conda activate striatum-2choice
$ conda install -c conda-forge cython numpy pandas scipy scikit-learn statsmodels h5py \
  tqdm scikit-image pillow matplotlib seaborn deprecated pytables opencv pip
$ pip install cmocean figurefirst
```

Git-clone the repository or download it as a zip file and unpack.

```bash
git clone https://github.com/wegmor/striatum-2choice.git
```

Navigate to the top of the repository folder structure and create a `data` subfolder.
Copy the data files `endoData_2019.hdf` and `alignment_190227.hdf` into the `data` folder.

Each figure is generated by its own python script. The figures will be saved to a subfolder `svg`.
Many of the figures require substantial, preparatory computation (12+ hours, e.g. due to 1,000 bootstrapping steps); the results of this preprocessing are stored in a `cache` folder as pickle files.

To generate any figure, activate the conda environment (`$ conda activate striatum-2choice`), navigate to the top of the repository folder structure, and run the associated python script (`$ python figureX.py`).
To see which script creates what figure, refer to the list below:

- **Fig. 1:**   `figure1OpenField.py`
- **Fig. 2:**   `figure2ChoiceIntro.py`
- **Fig. 3:**   `figure3Tunings.py`
- **Fig. 4:**   `figure4FollowNeurons.py`
- **Fig. 5:**   `figure5StaySwitchDecoding.py`
- **Fig. S2:**  `figureOpenFieldSupp.py`
- **Fig. S3:**  `figureContinuity.py`
- **Fig. S4:**  `figureTuningsSupp.py`
- **Fig. S5:**  `figureRewardSupp.py`
- **Fig. S6:**  `figureClusteringSupp.py`
- **Fig. S7:**  `figureDecodingSupp.py`
- **Fig. S8:**  `figureFollowNeuronsSupp.py`
- **Fig. S9:**  `figureStaySwitchDecodingSupp.py`
- **Fig. S10:** `figureQlearning.py`


### Software requirements

The code was tested with the following versions of the python packages used:

package       | version
--------------|--------------
python        | 3.8.3
cython        | 0.29.14
numpy         | 1.18.4
pandas        | 0.25.3
scipy         | 1.4.1
scikit-learn  | 0.22.1
statsmodels   | 0.11.0
h5py          | 2.9.0
tqdm          | 4.45.0
scikit-image  | 0.16.2
pillow        | 7.0.0
matplotlib    | 3.2.1
seaborn       | 0.9.0
deprecated    | 1.2.10
pytables      | 3.6.1
opencv        | 4.4.0
cmocean       | 2.0
figurefirst   | 0.0.6


### Reference

Weglage, M., Wärnberg, E., Lazaridis, I., Tzortzi, O., & Meletis, K. (2020). *Complete representation of action space and value in all striatal pathways.* bioRxiv, 2020.03.29.983825. https://doi.org/10.1101/2020.03.29.983825
