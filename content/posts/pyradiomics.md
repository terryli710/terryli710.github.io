---
title: Pyradiomics Simple Usage
date: 2020-09-11 14:58:49
tags: [python, package]
categories: NOTE
description: Brief for quick start in pyradiomics package
---

> Pyradiomics is an open-source python package for the extraction of radiomics data from medical images. Image loading and preprocessing (e.g. resampling and cropping) are first done using SimpleITK. Loaded data is then converted into numpy arrays for further calculation using multiple feature classes. Optional filters are also built-in.

## Ways to Deal with Medical Image Data

The reasons that we choose `pyradiomics` could be its openness, widely recognized and reasonably good performance with a variety of features available. Methods of analyzing medical images could be potential divided into two major ways:

1. Traditional Methods, which includes feature extraction and covert image data into another format which is more stable and universal;
2. Deep Methods, which involves using deep learning method to condense the image data into later layer's values. This methods can deal with images directly, but requires larger training data set.

Pyradiomics is absolutely an good option to use if using traditional methods.

## Pyradiomics Usage

### Simple Examples

#### Single  image/mask

The simplest way to use `pyradiomics` is through command line:

```python
pyradiomics <path/to/image> <path/to/segmentation> -o <save_path> -f csv
```

*NOTE*: 

- `pyradiomics` uses `simpleITK` to process raw image, so the file format of image and segmentation should be one of the formats that are readable by `simpleITK`:  .bmp, .pic, **.jpg**, .jpeg, **.tif**, .mha, **.nrrd**, .png, etc. (refer to [here](https://simpleitk.readthedocs.io/en/master/IO.html)). 
- Available [image types](https://pyradiomics.readthedocs.io/en/latest/customization.html#image-types) are also stored in `_enabledImageTypes` dictionary in the feature extractor class instance.
- `-o` parameter specifies the save path of extracted features, it should be a CSV file's path.
- `-f csv` specifies the format of saved file is CSV

#### Batch Mode

```bash
pyradiomics <path/to/input> -o <save_path> -f csv
```

*NOTE*: 

- Here, this `input` should be a CSV file that looks like this

| header of `path/to/image` | header of `path/to/seg` |
| ---------------------------- | ------------------------ |
| `path/to/image_1`           | `path/to/seg_1`          |
| `path/to/image_2`            | `path/to/seg_2`          |

- Batch mode provides a scalable way to apply `pyradiomics` feature extraction, since it needs only to organize a CSV file that contains information about files to extract features from and their segmentations, and one command line execution, to extract features from different files at once.

### In Code Usage

To use `pyradiomics` inside code, with similar effect to simple image/mask usage, code would be like:

```python
from radiomics import featureextractor
extractor = featureextractor.RadiomicsFeatureExtractor()
features = extractor.execute(image_nrrd_file_path, segmentation_nrrd_file_path)
```

The extractor will return a list of features, instead of store them into some files.

## Customization of Features

### Parameter File

#### Include Parameter File

A `yaml` or `JSON` format file could be provided to change the setting of feature class, as well as image type and **settings**. To use it in command line mode, like this:

```shell
pyradiomics <path/to/input> --param <parameter_file_path> -o <save_path> -f csv
```

And there two ways to add it in python codes:

```python
# add parameter file when creating instance
extractor = featureextractor.RadiomicsFeatureExtractor(param_file_path)
```

```python
# add parameter file after definition, using loadParams()
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.loadParams(param_file_path)
```

Note that only file path should be provided, `pyradiomics` will read in the file.

#### Customizations

Below is an example parameter file named `features.yaml`:

```yaml
imageType:
  Original: {}
  LoG:
    sigma: [0.5, 1.0, 2.0] 
  Wavelet: {}
  LBP3D:
    binWidth: 1.0
  Square: {}
  #SquareRoot: {}
  Logarithm: {}
  #Exponential: {}
  Gradient: {}
  #LBP2D: {}

featureClass:
  # redundant Compactness 1, Compactness 2 an Spherical Disproportion features are disabled by default, they can be
  # enabled by specifying individual feature names (as is done for glcm) and including them in the list.
  shape:
  firstorder:
  glcm:  # Disable SumAverage by specifying all other GLCM features available
    - 'Autocorrelation'
    - 'JointAverage'
    - 'ClusterProminence'
    - 'ClusterShade'
    - 'ClusterTendency'
    - 'Contrast'
    - 'Correlation'
    - 'DifferenceAverage'
    - 'DifferenceEntropy'
    - 'DifferenceVariance'
    - 'JointEnergy'
    - 'JointEntropy'
    - 'Imc1'
    - 'Imc2'
    - 'Idm'
    - 'Idmn'
    - 'Id'
    - 'Idn'
    - 'InverseVariance'
    - 'MaximumProbability'
    - 'SumEntropy'
    - 'SumSquares'
  glrlm:
  glszm:
  gldm:
  ngtdm: []
setting:
  # Normalization:
  # most likely not needed, CT gray values reflect absolute world values (HU) and should be comparable between scanners.
  # If analyzing using different scanners / vendors, check if the extracted features are correlated to the scanner used.
  # If so, consider enabling normalization by uncommenting settings below:
  #normalize: true
  #normalizeScale: 500  # This allows you to use more or less the same bin width.

  # Resampling:
  # Usual spacing for CT is often close to 1 or 2 mm, if very large slice thickness is used,
  # increase the resampled spacing.
  # On a side note: increasing the resampled spacing forces PyRadiomics to look at more coarse textures, which may or
  # may not increase accuracy and stability of your extracted features.
  interpolator: 'sitkBSpline'
  resampledPixelSpacing:
  #padDistance: 10  # Extra padding for large sigma valued LoG filtered images

  # Mask validation:
  # correctMask and geometryTolerance are not needed, as both image and mask are resampled, if you expect very small
  # masks, consider to enable a size constraint by uncommenting settings below:
  #minimumROIDimensions: 2
  #minimumROISize: 50

  # Image discretization:
  # The ideal number of bins is somewhere in the order of 16-128 bins. A possible way to define a good binwidt is to
  # extract firstorder:Range from the dataset to analyze, and choose a binwidth so, that range/binwidth remains approximately
  # in this range of bins.
  binWidth: 25

  # first order specific settings:
  voxelArrayShift: 1000  # Minimum value in HU is -1000, shift +1000 to prevent negative values from being squared.

  # Misc:
  # default label value. Labels can also be defined in the call to featureextractor.execute, as a commandline argument,
  # or in a column "Label" in the input csv (batchprocessing)
  label: 1
```

This file has three major parts: 

- [imageType](https://pyradiomics.readthedocs.io/en/latest/customization.html#image-types): image type to calculate features on. <value> is custom kwarg settings (dictionary). if <value> is an empty dictionary (‘{}’), no custom settings are added for this input image.
- [featureClass](https://pyradiomics.readthedocs.io/en/latest/customization.html#enabled-features): Feature class to enable, <value> is list of strings representing enabled features. If no <value> is specified or <value> is an empty list (‘[]’), all features for this class are enabled.
- [setting](https://pyradiomics.readthedocs.io/en/latest/customization.html#settings): Setting to use for pre processing and class specific settings. if no <value> is specified, the value for this setting is set to None.

Another potential part that is not displayed in this file is voxelSetting, which is used to control the voxel-based specific settings. Refer to the links for further information about the customizations that we can specify.

Further usage of `pyradiomics` could (hypothetically) be summarized in further blog articles.



