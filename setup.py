from pathlib import Path
from setuptools import setup, find_packages


def read_requirements(requirements_path: Path):
    requirements = []
    for line in requirements_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements

setup(
    name="splice",
    version="1.0",
    description="",
    author="Alex Oesterling, Usha Bhalla",
    author_email="aoesterling@g.harvard.edu, usha_bhalla@g.harvard.edu",
    py_modules=["splice"],
    packages=find_packages(exclude=["experiments*", "data*"]),
    install_requires=read_requirements(Path(__file__).with_name("requirements.txt")),
)
