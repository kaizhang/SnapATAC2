from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="snapatac2",
    description='SnapATAC: Single Nucleus Analysis Pipeline for ATAC-seq',
    url='', 
    author='Kai Zhang',
    author_email='kai@kzhang.org',
    license='MIT',
    version="0.1.0",
    rust_extensions=[RustExtension("snapatac2.snapatac2", binding=Binding.PyO3)],
    packages=["snapatac2"],
    zip_safe=False,
    python_requires=">=3.6",
    install_requires=[
        'numpy>=1.16.0',
        "anndata>=0.7.7",
        "scipy>=1.4",
        "umap-learn>=0.3.10",
    ],
)