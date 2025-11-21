import os

from setuptools import setup, find_packages

REQUIREMENTS = os.path.join(os.path.dirname(__file__), 'requirements.txt')

with open(REQUIREMENTS, 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='java_context_analyzer',
    version='0.1.0',
    packages=find_packages(),
    package_data={
        'multilspy.language_servers.eclipse_jdtls': ['static/**/*', '*.json'],
    },
    install_requires=requirements,
)
