# Twomes-Twutility: Inverse grey-box modelling and analysis tools for homes and utility buildings
This repository contains source code for the Twomes and Twutility digital twin heat balance models and inverse-grey-box analysis tools that support automated estimation of crucial parameters of buildings, installations and comfort needs in individual homes and utility buildings based on time-series monitoring data.

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
\<to be described soon\>

## Deploying
After cloning this this respository, install requirements: open a terminal windows (in JuypyterLabs availabe via the Launcher which itself is always avalable via the `+` tab), navigate to the root folder of this repository and enter this command: 
  ```shell
  pip install -r requirements.txt
  ```  

## Developing
This section describes how you can change the source code using a development environment and compile the source code into a service that can be deployed via the method described in the section [Deploying](#deploying).

### Prerequisites

If you want to run the code locally,you need:  
- [PyCharm](https://www.jetbrains.com/pycharm/download/#section=windows) Community or Professional Edition IDE installed (recommended)  
- [Python](https://www.python.org/downloads/) installed (min version 3.8)  

Start PyCharm and install the following packages by entering the following commands in a Terminal window in PyCharm:
- [Gekko](https://gekko.readthedocs.io/en/latest/) which can be installed using following command:
	```shell
	pip install gekko
	```  
- [numpy](https://numpy.org/install/) (min version 1.20.3) which can be installed by either following commands:<br/>
	```shell
	pip install numpy
	```
- [pandas](https://pandas.pydata.org/docs/getting_started/install.html) (min version 1.3.2) which can be installed by following commands:<br/>
	```shell
	pip install pandas
	```
- [matplotlib](https://matplotlib.org/stable/users/installing/index.html) (min version 3.4.2) which can be installed by following commands:<br/>
	```shell
	pip install matplotlib
	``` 
If you want to run the code on a JuluterLab server,you need:  
\<to be described soon\>

## Features
List of features ready and TODOs for future development. Ready:
* data extraction;
* measurement outlier removal and interpolation;
* GEKKO model with model parameters for biulding envelope.
* Analysis code to learn building model parameters;

To-do:
* extend GEKKO model with installation model parameters;
* refine measurement outlier removal via time series outlier reoval procedures.

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

