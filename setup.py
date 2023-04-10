from distutils.core import setup

from setuptools import find_packages
import os 

folder = os.path.dirname(os.path.realpath(__file__))
install_requires = [] # Here we'll get: ["gunicorn", "docutils>=0.3", "lxml==0.5a7"]
try:
    requirementPath = os.path.join(folder,'requirements.txt')
    if os.path.isfile(requirementPath):
        with open(requirementPath) as f:
            install_requires = f.read().splitlines()
except Exception as e:
    pass

setup(
    name="logic",
    version="1.0",
    packages=find_packages(),
    package_data={p: ["*"] for p in find_packages()},
    url="",
    license="",
    install_requires=install_requires,
    python_requires=">=3.7.1",
    author="ndams55",
    description="fraud-detection-in-electricity-and-gas-consumption-challenge",
)