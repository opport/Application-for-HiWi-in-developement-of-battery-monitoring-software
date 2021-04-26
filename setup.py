import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
	LONG_DESCRIPTION = fh.read()
with open("requirements.txt", "r") as fh:
	REQUIREMENTS = fh.readlines()
with open("LICENSE", "r") as fh:
	LICENSE = fh.read().strip()
with open("VERSION", "r") as fh:
	VERSION = fh.read().strip()

setup(
	name='Application-for-HiWi-in-developement-of-battery-monitoring-software',
	version=VERSION,
	packages=setuptools.find_packages(),
	url='https://github.com/ThanujSingaravelan/Application-for-HiWi-in-developement-of-battery-monitoring-software',
	license=LICENSE,
	install_requires=REQUIREMENTS,
	long_description=LONG_DESCRIPTION,
	author='Thanuj Singeravelan',
	author_email='<47354222+thanujsingaravelan@users.noreply.github.com>',
	description='Install script for task.py environment'
)
