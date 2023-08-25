from distutils.core import setup
from setuptools import find_packages

setup(
    name='ipk_mbpo',
    packages=find_packages(),
    version='1.0.0',
    description='Efficient reinforcement learning control for continuum robots based on Inexplicit Prior Knowledge',
    long_description=open('./README.md').read(),
    author='JunjiaLiu',
    author_email='junjialiu@sjtu.edu,cn',
    url='https://github.com/Skylark0924/TendonTrack',
    entry_points={
        'console_scripts': (
            'ipk_mbpo=softlearning.scripts.console_scripts:main',
            'viskit=mbpo.scripts.console_scripts:main'
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
