from setuptools import setup, find_packages

setup(

    name='wildfire_detector',
    version='0.1',
    packages=find_packages(),

    install_requires={
        'numpy',
        'opencv-python',
        'torch',
        'torchvision',
        'pyyaml',
        'Pillow',
    },

    include_package_data=True,

    package_data={
        'wildfire_detector': ['*.yaml', '*.pt']
    },
)