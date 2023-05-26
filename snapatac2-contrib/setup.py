from setuptools import setup

from pathlib import Path

ROOT_DIR = Path(__file__).parent
README = (ROOT_DIR / "README.md").read_text()

VERSION = {}
with open(ROOT_DIR / "snapatac2_contrib/_version.py") as fp:
    exec(fp.read(), VERSION)

setup(
    name="snapatac2-contrib",
    description='SnapATAC2: Single-cell epigenomics analysis pipeline',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://kzhang.org/SnapATAC2/', 
    license='MIT',
    version=VERSION['__version__'],
    packages=[
        "snapatac2_contrib",
        "snapatac2_contrib.metrics",
    ],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "snapatac2>=2.3.0",
        "anndata>=0.8.0",
        "numpy>=1.16.0",
        "pandas",
        "polars>=0.14.0, <=0.18.0",
        "scipy>=1.4",
        "typing_extensions",
    ],
)
