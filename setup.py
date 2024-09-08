from setuptools import setup,find_packages
from typing import List


wrong ="-e ."
def get_requirements(filepath:str)->List[str]:
    
    """
    to install all the requirements
    """
    requirements=[]
    with open(filepath) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if wrong in requirements:
            requirements.remove(wrong)




setup(
    name="ml-project1",
    version="0.0.0.1",
    author="cns",
    email="cheerlanithinsai8230@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)