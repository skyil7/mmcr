import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mmcr",
    version="0.1.0",
    author="Gio Paik",
    description="MMCR Loss: Learning efficient coding of natural images with maximum manifold capacity representations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/skyil7/mmcr",
    install_requires=[
        "einops",
    ],
    keywords=["torch", "loss", "mmcr", "Maximum Manifold Capacity Representations"],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
