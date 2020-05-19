# Lightning McQueen

Lighting McQueen is our phases of models created for arm gesture recognition. Bellow you will find
the setup process for beeing able to run the code used for machine learning part of this project.

## Requirements
* Python 3.6 or 3.7

* We recommend using PyCharm IDE (https://www.jetbrains.com/pycharm/)

## Configure and activate virtual  environment﻿
### PyCharm
If a virtual environment﻿ was not created when setting up a new project in PyCharm follow JetBrains guide on setting up a
virtual environment (https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)

### Terminal
#### Check to see if your install of Python has pip
```bash
pip -h
```
#### Install virtualenv package
```bash
pip install virtualenv
```

#### Create virtual environment
Make sure to be in the project directory.
```bash
virtualenv venv
```

#### Activate virtual environment
Mac OS and Linux
```bash
source venv/bin/activate
```
Windows
```bash
venv\Scripts\activate
```

If you want to deactivate the virtual environment simply run ```deactivate``` in the terminal window.

## Installation of needed packages
Use the package manager [pip](https://pip.pypa.io/en/stable/) for installing needed packages.

```bash
pip install plaidml-keras plaidbench
pip install -U matplotlib
pip install opencv-python
pip install tqdm
pip install termcolor
pip install sty
pip install pickle
```
### Setup PlaidML
```diff
+ If you have a dedicated GPU its recomended to select it as your accelerator.
```
Choose which accelerator you'd like to use (many computers, especially laptops, have multiple)
In the terminal of your python project (venv) write:

```bash
plaidml-setup
```

* Enable experimental mode
* Select your accelerator

Now try benchmarking MobileNet:
```bash
plaidbench keras mobilenet
```
* You are now good to go!
