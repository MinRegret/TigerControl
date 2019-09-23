# TigerControl
**********

**TigerControl** (Control and Time-Series Benchmarks) is an open-source framework for benchmarking control and time-series algorithms in simulated and real settings, and is available for anyone to download and use. By reducing algorithms to a set of standard APIs, TigerControl allows the user to quickly switch between models and tasks while running experiments and plotting results on the go, and for quick and simple comparison between model performances. TigerControl also comes with built-in standard control and time-series algorithms for comparison or other use.


Overview
========

Although there are several machine learning platforms that aid with the implementation of algorithms, there are far fewer readily available tools for benchmarks and comparisons. The main implementation frameworks (eg. Keras, PyTorch) provide certain small-scale tests, but these are in general heavily biased towards the batch setting. Other platforms such as OpenAI Gym are great for the testing of reinforcement learning methods, but as many online algorithms depend on access to interal dynamics or loss derivatives, this too is suboptimal. TigerControl exists to fill this gap — we provide a variety of evaluation settings to test online control and time-series algorithms on, with more diversity and generality than any other online benchmarking platform.


Installation
============

### Quick install (excluding PyBullet)

Clone the directory and use pip or pip3 to set up a minimal installation of TigerControl, which excludes PyBullet control problems.

```
    git clone https://github.com/johnhallman/tigercontrol.git
    pip install -e tigercontrol
```

You can now use TigerControl in your Python code by calling `import tigercontrol` in your files. 


### Full install (including PyBullet)

Before installing TigerControl's dependencies, create a new conda environment in order to avoid package conflicts.

```
    conda create -n name-of-your-environment python=3.x
    conda activate name-of-your-environment
```

Next, clone the GitHub repository and install the full package.

```
    git clone https://github.com/johnhallman/tigercontrol.git
    pip install -e 'tigercontrol[all]'
```

Finally, run a demo to verify that the installation was successful!

```
    python tigercontrol/problems/control/tests/test_minitaur.py
```


For more information
====================

To learn more about TigerControl and how to incorporate it into your research, check out the Quickstart guide in the ```tigercontrol/tutorials``` folder. Alternatively, check out our [readthedocs](https://tigercontrol.readthedocs.io/en/latest/) page for more documentation and APIs.

