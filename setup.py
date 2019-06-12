# sets up package

import setuptools

setuptools.setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='ctsb',
    url='https://github.com/johnhallman/ctsb',
    author='John Hallman',
    author_email='johnolof@gmail.com',
    # Needed to actually package something
    packages=setuptools.find_packages(),
    # *strongly* suggested for sharing
    version='1.0',
    # The license can be anything you like
    license='MIT',
    description='A skeleton implementation of the CTSB Python package',
    # We will also need a readme eventually (there will be a warning)
    long_description=open('README.md').read(),
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
)