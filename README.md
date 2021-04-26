# Application-for-HiWi-in-developement-of-battery-monitoring-software
Temperature task for application for HiWi in developement of battery monitoring software

# Setup

## Requirements
- `python3`: [Official Page](https://www.python.org/downloads/)
- `python3 pip`: [Official Documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/)
- A python3 dependency manager. This example uses `virtualenv`: [virtualenv installation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv)

## Installation and execution

This `virtualenv` example is a Linux example. For `virtualenv` commands on other OS examples, please refer to [this part of the documentation](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/#installing-virtualenv)  

Clone (download) and extract the repository, then run the following commands from within the directory:
```bash

# Change directory to project folder
cd <directory_path/Application-for-HiWi...>


# Create a virtualenv
virtualenv .venv
# Alternative command, if virtualenv is not in PATH
python3 -m virtualenv .venv

# Activate the virtualenv in your shell (assuming it's bash)
. .venv/bin/activate

# Install required dependencies in .venv
pip3 install -r requirements.txt

# Run program
python3 task.py

# exit .venv
deactivate
```

# Distribution build

[Official documentation](https://docs.python.domainunion.de/3/distutils/builtdist.html)

Using setuptools script `setup.py`:
```bash
# Within .venv virtual environment
python3 setup.py bdist
```
