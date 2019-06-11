CTSB
**********

CTSB (Control and Time-Series Benchmarks) is a framework for benchmarking and testing your control, time-series, and to some extent, general online algorithms in simulated and real settings.** The ``ctsb`` open-source library is available for anyone to download and use.


Overview
========

Although there are several machine learning platforms that aid with the implementation of algorithms, there are far fewer readily available tools for benchmarks and comparisons. The main implementation frameworks (eg. Keras, PyTorch) provide certain small-scale tests, but these are in general heavily biased towards the batch setting. Other platforms such as OpenAI Gym are great for the testing of reinforcement learning methods, but as many online algorithms depend on access to interal dynamics or loss derivatives, this too is suboptimal. CTSB exists to fill this gap — we provide a variety of evaluation settings to test online control and time-series algorithms on, with more diversity and generality than any other online benchmarking platform.


Installation
============

Either install ``ctsb`` by cloning this GitHub repo:

.. code:: shell

    git clone https://github.com/johnhallman/ctsb.git
    cd ctsb
    pip install -e .

Or install the packaged directly from PyPI:

.. code:: shell

    pip install ctsb