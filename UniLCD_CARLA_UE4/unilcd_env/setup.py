from setuptools import setup, find_packages

setup(
    name='unilcd_env',
    version='1.0.0',
    packages=find_packages(include=['unilcd_env', 'unilcd_env.*']),
    install_requires=['gymnasium>=0.28.1','pygame>=2.1.2']
)