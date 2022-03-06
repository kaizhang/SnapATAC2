from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="snapatac2",
    description='SnapATAC: Single Nucleus Analysis Pipeline for ATAC-seq',
    url='https://kzhang.org/SnapATAC2/', 
    author='Kai Zhang',
    author_email='kai@kzhang.org',
    license='MIT',
    version="1.99.99.3",
    rust_extensions=[RustExtension("snapatac2._snapatac2", binding=Binding.PyO3)],
    packages=[
        "snapatac2",
        "snapatac2.preprocessing",
        "snapatac2.tools",
        "snapatac2.plotting",
    ],
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "anndata>=0.8.0rc1",
        "numpy>=1.16.0",
        "pandas",
        "python-igraph",
        "scipy>=1.4",
        "scikit-learn>=0.22",
        "tqdm>=4.62",
        "umap-learn>=0.3.10",
        "plotly>=5.6.0"
    ],
)
