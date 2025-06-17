from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="NiChart_DLWMLS",
    version="0.0.1",
    description="Run Deep-Learning-based-White-Matter-Lesion-Segmentation on your data (requires FLAIR, optional T1 masks for granular segmentation).",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kyunglok Baik",
    author_email="software@cbica.upenn.edu",
    maintainer="Kyunglok Baik, Alexander Getka",
    license="By installing/using NiChart_DLWMLS, the user agrees to the following license: See https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html",
    url="https://github.com/CBICA/NiChart_DLWMLS",
    python_requires=">=3.9",
    install_requires=[
        "DLWMLS",
        "huggingface_hub",
        "pandas",
        "scipy",
        "nibabel",
        "argparse",
        "pathlib",
    ],
    # entry_points={"console_scripts": ["NiChart_DLWMLS = NiChart_DLWMLS.__main__:run_full",
    #                                   "NiChart_DLWMLS_essential = NiChart_DLWMLS.__main__.run_essential"]},
    entry_points={"console_scripts": ["NiChart_DLWMLS = NiChart_DLWMLS.__main__:main"]},
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Operating System :: Unix",
    ],
    packages=find_packages(exclude=[".github"]),
    include_package_data=True,
    keywords=[
        "deep learning",
        "image segmentation",
        "semantic segmentation",
        "medical image analysis",
        "medical image segmentation",
        "nnU-Net",
        "nnunet",
    ],
    package_data={
        "NiChart_DLWMLS": ["**/*.csv", "**/*.json"],
    },
)