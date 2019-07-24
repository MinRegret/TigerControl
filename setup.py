# sets up package

import setuptools

# 0.xx = alpha, 1.xx = beta, >2.xx = full release
version = '0.1'

setuptools.setup(
    name='ctsb',
    url='https://github.com/johnhallman/ctsb',
    author='John Hallman',
    author_email='johnolof@gmail.com',
    packages=setuptools.find_packages(),
    install_requires=open('requirements.txt').read().split('\n'),
    extras_require={
        'all': ['pybullet', 'gym'],
            },
    version=version,
    license='Apache License 2.0',
    description='A control and time-series algorithms benchmarking framework',
    long_description=open('README.md').read(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: Apache License 2.0",
		"Operating System :: OS Independent",
	],
)