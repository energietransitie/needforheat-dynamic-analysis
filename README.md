# Twomes inverse grey-box modelling and analysis tools for homes and utility buildings
This repository contains source code for the Twomes digital twin heat balance models and inverse-grey-box analysis tools, based on [GEKKO Python](https://github.com/BYU-PRISM/GEKKO). This analysis software can be regarded as a particular form of physics informed machine learning for automated estimation of crucial parameters of buildings, installations and comfort needs in individual homes and utility buildings based on time-series monitoring data.

## Table of contents
* [General info](#general-info)
* [Prerequisites](#prerequisites)
* [Deploying](#deploying)
* [Developing](#developing) 
* [Features](#features)
* [Status](#status)
* [License](#license)
* [Credits](#credits)

## General info

This repository contains the GEKKO Python-based implementation of inverse grey-box analysis software. The purpose of this sofware is to (help) speedup the energy transition, in particular the heating transition. 

We developed this software in the [Twomes](https://edu.nl/9fv8w) project (to learn building parameters) and the [Brains4Buildings](https://edu.nl/kynxd) project (to learn from the relation between occupancy, ventilation rates and CO<sub>2</sub> concentration). This reposotory contains data for virtual homes and virtual rooms that were used to verify the proper implementation of the GEKKO models. 

The data we collected, including a metadata can be found in these related repositories:
- [twomes-dataset-assendorp2021](https://github.com/energietransitie/twomes-dataset-assendorp2021).
- [brains4buildings-dataset-windesheim2022](https://github.com/energietransitie/brains4buildings-dataset-windesheim2022).

## Prerequisites

Both for deplopying of and developing for the software in this repository, you can use a [JupyterLab](https://jupyter.org/) environment on your local machine. Other environments, such as [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) and [Visual Studio Code](https://code.visualstudio.com/) may work as well, but we do not include documentation for this here. 

**Note**
As an alternative, you can also [install and use JupyterLab in a docker container on a server](https://github.com/energietransitie/twomes-backoffice-configuration#jupyterlab).

To use JupyterLab on your local machine, make sure you have the following software properly installed and configured on your machine:

### Step 1: Install Python
If you haven't already installed it, go to [Python](https://www.python.org/downloads/) and install at least version 3.8, which comes pre-installed with `pip`, the package manager you need in the steps below.     

### Step 2: Install and Launch JupyterLab

If you haven't already installed JupyterLab, you can [install JupyterLab](https://jupyter.org/install#jupyterlabl) with the following pip command in your terminal:

```
pip install jupyterlab
```

To add support for git from within the JupyterLab environment, issue the following command in your terminal:
```
pip install jupyterlab-git
```

Once you've installed JupyterLab, you can launch JupyterLab by running the following command in your terminal:

```
jupyter-lab
```

This will open JupyterLab in your default web browser. 
### Step 3: Clone this Repository

In JupyterLab, navigate to the folder where you would like to clone this repository, select the git-icon in the left pane, select `Clone a Repository` and pase the URI for this repository, which is available via the green `<> Code` button on the GitHub page of this repository.

### Step 4: Install the Required Dependencies

After cloning this this respository, install requirements: open a terminal in JuypyterLab (available via the JupyterLab Launcher, via the `+` tab), navigate to the root folder of your clone of this repository and enter this command: 
  ```shell
  pip install -r requirements.txt
  ```  

This will install all the required dependencies listed in the [`requirements.txt`](requirements.txt) file.

## Deploying

This section describes how you can use the IPython notebooks, without changing the Python code. After installing JupyterLab as described above, you can run the software by opening up `.ipynb ` files and run the contents from the [`/example/`](https://github.com/energietransitie/twomes-inverse-grey-box-analysis/tree/update-readme-for-public-review/examples) folder. We've created example files based on or work in multiple projects:

- `<Project>ExtractionBackup.ipynb` files contain code you can run to extract measurement data from a Twomes server and save it as [parquet](https://parquet.apache.org/) files. These .ipynb files only work when you run the code in a JupyterLab environment that has [access](https://github.com/energietransitie/twomes-backoffice-configuration#twomes_db_url-1) to the [MariaDB database](https://github.com/energietransitie/twomes-backoffice-configuration#twomes_db_url-1) on a [Twomes backoffice server](https://github.com/energietransitie/twomes-backoffice-configuration).
- `<Project>_to_CSV.ipynb` files contain code you can run to convert a parquet file containing DataFrames to multiple [zip](https://en.wikipedia.org/wiki/ZIP_(file_format))ped [csv](https://en.wikipedia.org/wiki/Comma-separated_values) files, a single file containing all measurements and one zipped csv file per id. Parquet files load faster and are smaller than zipped csv files. Nevertheless, for backward compatibility with data analytics tools that are not yet able to process parquet files, we used the code in these .ipynb files to create the contents for the open data repositories. You can find ore information about the formatting of DataFrames with measurements and DataFrames with properties, as well as the open data itself in the repositories [twomes-dataset-assendorp2021](https://github.com/energietransitie/twomes-dataset-assendorp2021) and [brains4buildings-dataset-windesheim2022](https://github.com/energietransitie/brains4buildings-dataset-windesheim2022). 
- `<Project>_analysis_virtual_ds.ipynb` files contain code you can run to verify whether a mathematical model is properly implemented in GEKKO code the functions `learn_home_parameters()` or `learn_room_parameters()` from [`analysis\inversegreyboxmodel.py`](https://github.com/energietransitie/twomes-inverse-grey-box-analysis/blob/update-readme-for-public-review/analysis/inversegreyboxmodel.py). To perform the validation, we created 'virtual data', i.e. time series data for a virtual home or virtual room that behaves exactly according to the the mathematical model and has no measurement errors nor measurement hickups. This 'virtual data' was generated using an Excel implementation of the same mathematical model. You can find the virtual data in the `/data/<project>_virtual_ds/` folders.
- `<Project>_PlotTest.ipynb` files contain example code that demonstrate the various ways you can plot (parts of) a DataFrame contraining properties or preprocessed data, using the functions `dataframe_properties_plot()` and `dataframe_preprocessed_plot()`, respectively, from [`view\plotter.py`]([view\plotter.py]).
- `<Project>_analysis_real_ds.ipynb` files contain the functions `learn_home_parameters()` or `learn_room_parameters()` from [`analysis\inversegreyboxmodel.py`](analysis\inversegreyboxmodel.py) to perform grey-box analysis, on datasets with real measurements. Currently, we set up these analysis functions to perform various steps:
	- Read the parquet files from `<Project>ExtractionBackup.ipynb`, which contains a properties DataFrame.
	- Preprocess the data to make it suitable for analysis, which involves both outlier removal and time-based interpolation and which ultimately results in a preprocessed DataFrame. 
	- Perform the analysis over consecutive learning periods, e.g. 7 days or 3 days, resulting in both a DataFrame with learned variables and error metrics per id per learning period and a DataFrame with the resulting optimal time series for the property used as the fitting objective and the values of any learned time-dependent properties.
	- Visualize the analysis results in graphs.

## Developing
This section describes how you can change the source code. You can do this using JupyterLab, as described in the section [Deploying](#deploying). Other development environments, such as [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) and [Visual Studio Code](https://code.visualstudio.com/) may work as well, but we do not include documentation for this here. 

Should you find any issues or bugs in our code, please report them via the [issues](https://github.com/energietransitie/twomes-inverse-grey-box-analysis/issues) tab of this repository.

To change the code, we recommend:
- Try out your changes using the various `.ipynb ` files from the [`/example/`](/example/) folder. The section [Deploying](#deploying) contains a high level description of these files.
- Migrate stable code to functions in Python files.
- Should you have extensions or bug fixes that could be useful for other users of the repository as well, please fork this reposotory and make a Pull Request on this repository. 

## Features
Features include:
* data extraction;
* data preprocessing: measurement outlier removal and interpolation;
* `learn_home_parameters()` function in [analysis\inversegreyboxmodel.py](analysis\inversegreyboxmodel.py) that uses a GEKKO model and code to learn building model parameters such as specific heat loss [W/K], thermal intertia [h], thermal ass [Wh/K] and apparent solar aperture [m<sup>2</sup>] of a building;
* `learn_room_parameters()` function in [analysis\inversegreyboxmodel.py](analysis\inversegreyboxmodel.py) that uses a GEKKO model and code to learn:
	* apparent infiltration area [m<sup>2</sup>] and ventilation rates [m<sup>3</sup>/h] from CO<sub>2</sub> concentration [ppm] and occupancy [p] time series data;
	* apparent infiltration area [m<sup>2</sup>] and occupancy [p] from from CO<sub>2</sub> concentration [ppm] and ventilation rates [m<sup>3</sup>/h]  time series data;

To-do:
* update code in the `learn_home_parameters()` function to align with the newer code and preprocessing tools used in the `learn_room_parameters()` function;
* extend GEKKO model in `learn_home_parameters()` with installation model details to learn installation parameters;
* add 'dynamic' measurement outlier removal for measurement time series before interpolation, i.e. a rolling window outlier removal procedure, similar to a [hampel filter](https://pypi.org/project/hampel/) but working on non-equidistant time-series data and using a duration as a time window;
* combine the models in `learn_home_parameters()` and `learn_room_parameters()` and apply on a suitable dataset to figure out whether adding CO<sub>2</sub> concentration and occupancy [p] time series data helps to learn ventilation losses and other heat losses separately.
* add time series data about wind to the model and figure out whether (wind-induced) infiltration losses, ventilation losses and other heat losses of a building can be learned separately. 

## Status
Project is: _in progress_

## License
This software is available under the [Apache 2.0 license](./LICENSE), Copyright 2021 [Research group Energy Transition, Windesheim University of Applied Sciences](https://windesheim.nl/energietransitie) 

## Credits
This software is a collaborative effort of:
* Hossein Rahmani · [@HosseinRahmani64](https://github.com/HosseinRahmani64)
* Henri ter Hofte · [@henriterhofte](https://github.com/henriterhofte) · Twitter [@HeNRGi](https://twitter.com/HeNRGi)

It is partially based on earlier work by the following students:
* Casper Bloemendaal · [@Bloemendaal](https://github.com/Bloemendaal)
* Briyan Kleijn · [@BriyanKleijn](https://github.com/BriyanKleijn)
* Nathan Snippe · [@nsrid](https://github.com/nsrid)
* Jeroen Matser · [@Spudra](https://github.com/Spudra)
* Steven de Ronde · [@SteviosSDR](https://github.com/SteviosSDR)
* Joery Grolleman · [@joerygrolleman](https://github.com/joerygrolleman)

Product owner:
* Henri ter Hofte · [@henriterhofte](https://github.com/henriterhofte) · Twitter [@HeNRGi](https://twitter.com/HeNRGi)

We use and gratefully aknowlegde the efforts of the makers of the following source code and libraries:
* [GEKKO Python](https://github.com/BYU-PRISM/GEKKO), by Advanced Process Solutions, LLC., licensed under [an MIT-style licence](https://github.com/BYU-PRISM/GEKKO/blob/master/LICENSE)
* [Twomes Analysis Pipeline, v1](https://github.com/energietransitie/twomes-analysis-pipeline), by Research group Energy Transition, Windesheim University of Applied Sciences, licensed under [Apache-2.0 License](https://github.com/energietransitie/twomes-analysis-pipeline/blob/main/LICENSE)
* [HourlyHistoricWeather](https://github.com/stephanpcpeters/HourlyHistoricWeather), by [@stephanpcpeters](https://github.com/stephanpcpeters), licensed under [an MIT-style licence](https://raw.githubusercontent.com/stephanpcpeters/HourlyHistoricWeather/master/historicdutchweather/LICENSE)

