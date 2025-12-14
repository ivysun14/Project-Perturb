from setuptools import setup, find_packages

setup(
    name="perturbseq-lang",
    version="0.1.0",
    description="Tools for processing Perturb-seq data for LLM fine-tuning and evaluation",
    author="Chen Lab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "openai": ["openai>=1.0.0"],
        "viz": ["matplotlib>=3.5.0", "seaborn>=0.11.0"],
        "dev": ["pytest>=7.0.0", "black", "flake8"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
