from setuptools import find_packages,setup
from typing import List

from typing import List

def get_requirements() -> List[str]:
    """
    This function reads a 'requirements.txt' file and returns a list of requirements,
    excluding editable installs (lines starting with '-e') and empty lines.
    """
    requirement_list: List[str] = []

    try:
        with open('requirements.txt', 'r') as file:
            # Read all lines from the file
            lines = file.readlines()

            # Process each line
            for line in lines:
                requirement = line.strip()

                # Skip empty lines and lines starting with '-e'
                if requirement and not requirement!= '-e .':
                    requirement_list.append(requirement)

    except FileNotFoundError:
        print("requirements.txt file not found")

    return requirement_list

setup(
    name="Project1",
    version="1.0.0",
    author="Preet Rank",
    author_email="preetrank53@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)