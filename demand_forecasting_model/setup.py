from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="demand-forecasting-model",
    version="1.0.0",
    author="Urban Black ML Team",
    description="Demand forecasting model for ride-sharing service",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "tensorflow>=2.14.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "pyyaml>=6.0.1",
        "python-dotenv>=1.0.0",
        "geohash2>=1.1.0",
    ],
)
