Installation
============

Install from Source
-------------------

Clone the repository and install in development mode:

.. code-block:: bash

   git clone https://github.com/Sean0418/connectopy.git
   cd connectopy
   pip install -e ".[dev,docs]"

Dependencies Only
-----------------

To install just the core dependencies:

.. code-block:: bash

   pip install -e .

Development Setup
-----------------

For development with linting, testing, and documentation tools:

.. code-block:: bash

   pip install -e ".[dev,docs]"
   pre-commit install

Running Tests
-------------

.. code-block:: bash

   pytest

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html

The documentation will be available at ``docs/_build/html/index.html``.
