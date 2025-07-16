from setuptools import setup, find_packages

setup(

    name='wildfire_detector',
    version='0.1.0',
    packages=find_packages(),

    install_requires=[
        "torch",  # for torch, torch.nn, torch.nn.functional
        "torchvision",  # for torchvision.transforms, models
        "Pillow",  # for PIL.Image
        "PyYAML",  # for yaml
        "requests",  # for HTTP requests
        "opencv-python",  # for cv2
        "numpy",  # for np
        "scikit-learn",  # for DBSCAN
        "matplotlib",  # for matplotlib.pyplot, patches
    ],

    include_package_data=True,

    package_data={
        'wildfire_detector': ['*.yaml', '*.pt']
    },
)