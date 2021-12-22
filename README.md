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
Add more general information about the repo. What is purpose of the code in the repo? Motivation?

## Prerequisites
Following packages are required to run the code locally:
[Python](https://www.python.org/downloads/) (min version 3.8)

[Gekko](https://gekko.readthedocs.io/en/latest/) which can be installed using following command: 
	```shell
	pip install gekko
	```

[numpy](https://numpy.org/install/) (min version 1.20.3) which can be installed by following commands:<br/>
pip:<br/>
	```shell
	pip install numpy
	```

conda:<br/>
	```shell
	conda install -c anaconda numpy
	```


[pandas](https://pandas.pydata.org/docs/getting_started/install.html) (min version 1.3.2) which can be installed by following commands:<br/>
pip:<br/>
	```shell
	pip install pandas
	```

conda:<br/>
	```shell
	conda install -c anaconda pandas
	```


[matplotlib](https://matplotlib.org/stable/users/installing/index.html) (min version 3.4.2) which can be installed by following commands:<br/>
pip:<br/>
	```shell
	pip install matplotlib
	```
 
conda:<br/>
	```shell
	conda install -c conda-forge matplotlib
	```

## Deploying
Describe how the reader can download and install the lastest installable version(s). If appropriate, link to the latest binary release or package you published in the repo. If needed, describe this for different platforms.
Use steps if the procedure is non-trivial:
1. first step;
2. second step;
3. final step.

Format any scripts or commands in a way that makes them  easy to copy, like the following example. 

Forgotten your Wi-Fi password? No problem with the follwing command, replacing `SSID` with the Wi-Fi name of your own Wi-Fi network: 
```shell
netsh wlan show profile SSID key=clear
```

## Developing
Describe how the reader can use / adapt/ compile the souce code. 

## Features
List of features ready and TODOs for future development. Ready:
* awesome feature 1;
* awesome feature 2;
* awesome feature 3.

To-do:
* wow improvement to be done 1;
* wow improvement to be done 2.

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

Thanks also go to:
* Stephan Peters · [@stephanpcpeters](https://github.com/stephanpcpeters) 

Product owner:
* Henri ter Hofte · [@henriterhofte](https://github.com/henriterhofte) · Twitter [@HeNRGi](https://twitter.com/HeNRGi)

We use and gratefully aknowlegde the efforts of the makers of the following source code and libraries:
* [GEKKO Python](https://github.com/BYU-PRISM/GEKKO), by Advanced Process Solutions, LLC., licensed under [an MIT-style licence](https://github.com/BYU-PRISM/GEKKO/blob/master/LICENSE)
* [Twomes Analysis Pipeline, v1](https://github.com/energietransitie/twomes-analysis-pipeline), by Research group Energy Transition, Windesheim University of Applied Sciences, licensed under [Apache-2.0 License](https://github.com/energietransitie/twomes-analysis-pipeline/blob/main/LICENSE)
