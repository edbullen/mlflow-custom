from setuptools import setup, find_packages


# Dependencies for the package
INSTALL_REQUIRES = [
      'numpy',
      'pandas',
      'mlflow'
]

setup(
    name='hybridfunction',
    install_requires=INSTALL_REQUIRES,
    #package_dir = {'': 'hybridfunction'},
    #packages=find_packages(include=["hybridfunction"]),
    packages=["hybridfunction"],
    version='1.0.3',
    author='Edward Bullen',
)
