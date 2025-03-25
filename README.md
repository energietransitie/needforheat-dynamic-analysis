# NeedForHeat Diagnosis: Physics-informed Machine Learning of Residential Heat Performance Signatures

This repository contains source code for the NeedForHeat dynamic heat balance and mass balance models and analysis tools, based on [GEKKO Python](https://github.com/BYU-PRISM/GEKKO). This analysis software is a form of physics-informed machine learning for automated estimation of crucial building, installation, and comfort parameters using time-series monitoring data.

## Table of Contents
* [General Info](#general-info)
* [Prerequisites](#prerequisites)
* [Deploying](#deploying)
* [Developing](#developing)
* [Features](#features)
* [Status](#status)
* [License](#license)
* [Credits](#credits)

## General Info

This repository contains the GEKKO Python-based implementation of physics-informed machine learning (formerly known as inverse grey-box analysis). The purpose of this software is to accelerate the heating transition as part of the broader energy transition.

This software was developed in the context of multiple projects:

* [Twomes](https://edu.nl/9fv8w): Our first research project aimed at building digital twins using inverse grey-box modeling to learn physical building parameters.
* [Brains4Buildings](https://edu.nl/kynxd): A research project exploring the relationship between occupancy, ventilation rates, and CO<sub>2</sub> concentration.
* [REDUCEDHEATCARB](https://edu.nl/gutuc): Our latest research project, extending previous models with:
  - Ventilation heat loss modeling
  - Wind-dependent infiltration heat loss modeling
  - A model separating heat generation (boiler/heat pump) from heat distribution (e.g., hydronic radiators)

Field data and metadata descriptions are available in related repositories:

* [twomes-dataset-assendorp2021](https://github.com/energietransitie/twomes-dataset-assendorp2021) (work in progress – data not yet available)
* [brains4buildings-dataset-windesheim2022](https://github.com/energietransitie/brains4buildings-dataset-windesheim2022)
* [needforheat-dataset-reducedheatcarb2023](https://github.com/energietransitie/needforheat-dataset-reducedheatcarb2023) (work in progress – data and metadata not yet available)


> **Note:** The previous Twomes and Brains4Buildings code has been deprecated and moved to the `twomes-brainsforbuildings-deprecated` branch. To access that version, please see the corresponding `README-deprecated.md` in that branch.

## Prerequisites

### Server-Based Installation
We tested this repository on a Linux server with the following characteristics:

* **OS:** Ubuntu 22.04.2 LTS, Kernel: 5.15.0-125-generic
* **CPU:** 16 vCPUs
* **Memory:** 58 GiB RAM, 99 GiB swap
* **Disk:** 1.9 TB

To set up a server, follow the guide at [NeedForHeat Server Configuration](https://github.com/energietransitie/needforheat-server-configuration). However, for a **NeedForHeat Diagnosis-only** server, you can **skip** the following components:

* NeedForHeat Server API (includes MariaDB)
* Manuals
* CloudBeaver
* Duplicati

Instead, ensure the following are installed:

* Portainer
* Traefik Proxy
* JupyterLab

### Local Installation
Alternatively, you can run the software on your local machine using [JupyterLab](https://jupyter.org/). To do so:

1. Install Python (>=3.8) from [python.org](https://www.python.org/downloads/)
2. Install JupyterLab using pip:
   ```shell
   pip install jupyterlab jupyterlab-git
   ```
3. Clone this repository and install dependencies:
   ```shell
   git clone <repository_url>
   cd <repository_folder>
   pip install -r requirements.txt
   ```
4. Launch JupyterLab:
   ```shell
   jupyter-lab
   ```

## Deploying

### Running Notebooks
After installing JupyterLab, you can execute `.ipynb` notebooks from the `examples/` folder. 

**Important:**
> Documenting the exact order in which notebooks should be run is **work in progress**. Updates will follow soon.

## Features

> Documenting the current features is **work in progress**. Updates will follow soon.

**Upcoming Updates:**
> The exact workflow for using these features efficiently is **work in progress**. More information will follow soon.

## Status
This repository documents the final state of the REDUCEDHEATCARB project and remains actively maintained.

## Status
Project is: _in progress_

## License
This software is available under the [Apache 2.0 license](/LICENSE), Copyright 2025 [Research group Energy Transition, Windesheim University of Applied Sciences](https://windesheim.nl/energietransitie) 

## Credits
Author:
* Henri ter Hofte · [@henriterhofte](https://github.com/henriterhofte) · Twitter [@HeNRGi](https://twitter.com/HeNRGi)

With contributions from:
* Hossein Rahmani · [@HosseinRahmani64](https://github.com/HosseinRahmani64)
* Ruben Cijsouw
* Ivo Gebhardt
* Carlos Mora Moreno
* Mathijs van de Weerd

It is partially based on earlier work by the following students:
* Casper Bloemendaal · [@Bloemendaal](https://github.com/Bloemendaal)
* Briyan Kleijn · [@BriyanKleijn](https://github.com/BriyanKleijn)
* Nathan Snippe · [@nsrid](https://github.com/nsrid)
* Jeroen Matser · [@Spudra](https://github.com/Spudra)
* Steven de Ronde · [@SteviosSDR](https://github.com/SteviosSDR)
* Joery Grolleman · [@joerygrolleman](https://github.com/joerygrolleman)

Product owner:
* Henri ter Hofte · [@henriterhofte](https://github.com/henriterhofte) · Twitter [@HeNRGi](https://twitter.com/HeNRGi)

We use and gratefully acknowlegde the efforts of the makers of the following source code and libraries:
* [GEKKO Python](https://github.com/BYU-PRISM/GEKKO), by Advanced Process Solutions, LLC., licensed under [an MIT-style licence](https://github.com/BYU-PRISM/GEKKO/blob/master/LICENSE)
* [Twomes Analysis Pipeline, v1](https://github.com/energietransitie/twomes-analysis-pipeline), by Research group Energy Transition, Windesheim University of Applied Sciences, licensed under [Apache-2.0 License](https://github.com/energietransitie/twomes-analysis-pipeline/blob/main/LICENSE)
* [HourlyHistoricWeather](https://github.com/stephanpcpeters/HourlyHistoricWeather), by [@stephanpcpeters](https://github.com/stephanpcpeters), licensed under [an MIT-style licence](https://raw.githubusercontent.com/stephanpcpeters/HourlyHistoricWeather/master/historicdutchweather/LICENSE)

