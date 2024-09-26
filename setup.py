from pathlib import Path

from setuptools import setup

auto_lirpa_path = (Path(__file__).parent / "auto_LiRPA").as_uri()

setup(
    install_requires=[
        f"auto_LiRPA @ {auto_lirpa_path}",
        "torch==1.12.1",
        "torchvision==0.13.1",
        "numpy>=1.25,<1.26",
        "scipy>=1.11,<1.12",
        "pandas==2.1.4",
        "seaborn==0.12.2",
        "frozendict>=2.3.8,<3.0",
        "rust_enum==1.1.5",
        "fairness-datasets==0.4.0",
        "folktables==0.0.12",
        "multiprocess==0.70.15",
        "scikit-learn>=1.3.2,<1.4.0",
        "kmodes==0.12.2",
        "optuna>=3.5.0,<4.0",
        "matplotlib>=3.8.2,<3.9.0",
        "requests>=2.31,<3.0",
        "randomname==0.2.1",
        "ruamel.yaml==0.18.6",
        "jupyter",
        "papermill>=2.6.0,<3.0",
        "psutil>=5.9.8,<6.0",
        "py-cpuinfo>=9.0.0,<10.0.0",
        "GPUtil>=1.4.0,<2.0.0",
        "pytest>=7.4,<7.5",
    ],
)
