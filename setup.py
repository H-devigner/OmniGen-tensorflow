from setuptools import setup, find_packages

setup(
    name="omnigen-tf",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.13.0",
        "transformers",
        "pillow",
        "numpy",
        "huggingface-hub"
    ],
    author="H-devigner",
    author_email="",
    description="TensorFlow implementation of OmniGen multi-modal generative AI model",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/H-devigner/OmniGen-tensorflow",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
