"""EuroSAT land-use classification: training, serving and operations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("eurosat-mlops")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)
IMAGE_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
