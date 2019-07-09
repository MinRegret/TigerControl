# CTSB
**********

**CTSB** (Control and Time-Series Benchmarks) is an open-source framework for benchmarking control and time-series algorithms in simulated and real settings, and is available for anyone to download and use. By reducing algorithms to a set of standard APIs, CTSB allows the user to quickly switch between models and tasks while running experiments and plotting results on the go, and for quick and simple comparison between model performances. CTSB also comes with built-in standard control and time-series algorithms for comparison or other use.


Overview
========

Although there are several machine learning platforms that aid with the implementation of algorithms, there are far fewer readily available tools for benchmarks and comparisons. The main implementation frameworks (eg. Keras, PyTorch) provide certain small-scale tests, but these are in general heavily biased towards the batch setting. Other platforms such as OpenAI Gym are great for the testing of reinforcement learning methods, but as many online algorithms depend on access to interal dynamics or loss derivatives, this too is suboptimal. CTSB exists to fill this gap — we provide a variety of evaluation settings to test online control and time-series algorithms on, with more diversity and generality than any other online benchmarking platform.


Installation
============

Before installing CTSB and its dependencies, create a new conda environment in order to avoid package conflicts.

```
    conda create -n name-of-your-environment python=3.x
    conda activate name-of-your-environment
```

Next, either install CTSB by cloning this GitHub repo if you want to make customizable changes...

```
    git clone https://github.com/johnhallman/ctsb.git
    cd ctsb
    pip install -e .
    cd ..
```

... or install the package directly using pip.

```
    pip install git+https://github.com/johnhallman/ctsb.git
```

Finally, run a demo to verify that the installation was successful!

```
    cd ctsb
    python ctsb/problems/tests/test_pendulum.py
```


Quickstart
============

To learn more about CTSB and how to incorporate it into your research, check out the Quickstart guide in the ```ctsb/notebooks``` folder.

