from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
with open('kinetix/requirements.txt') as f:
    required = f.read().splitlines()

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

# Get extra files
extra_files = package_files('kinetix/ci_data')
extra_files += package_files('kinetix/mechanisms')

setup(
    name='kinetix',
    version='0.1',
    packages=find_packages(include=['kinetix', 'kinetix.*']),
    include_package_data=True,
    package_data={
        'kinetix': extra_files,
    },
    install_requires=required,
    entry_points={
        'console_scripts': [
            'kinetix=kinetix.__main__:main',
        ],
    },
    # Metadata
    author='Bogdan Danciu',
    author_email='danciub@ethz.ch',
    description='A code generator for reaction kinetics, thermodynamics and transport properties',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/bogdandanciu/KinetiX',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
    ],
    python_requires='>=3.8',
)
