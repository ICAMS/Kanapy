.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/mrgprasad/kanapy/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Kanapy could always use more documentation, whether as part of the
official kanapy docs, in docstrings, or even on the web in blog posts,
articles, and alike.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/mrgprasad/kanapy/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.

Get Started!
------------

Ready to contribute? Here's how to set up `kanapy` for local development.

.. note:: For a complete tutorial on forking, see https://help.github.com/en/articles/fork-a-repo

1. Fork the `kanapy` repo on GitHub.
2. Clone your fork locally:

.. code-block:: console

    $ git clone git@github.com:your_name_here/kanapy.git

3. Install your local copy into a virtual environment. Assuming you have Anaconda installed, this is how you set up your fork for local development:

.. code-block:: console

    $ conda create -n knpy
    $ conda activate knpy
    (knpy) $ cd kanapy/

4. Create a branch for local development:

.. code-block:: console

   (knpy) $ git checkout -b name-of-your-bugfix-or-feature

   
   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass the tests:

.. code-block:: console
    
    (knpy) $ pytest tests/ -v

   
6. Commit your changes and push your branch to GitHub:

.. code-block:: console

    (knpy) $ git add .
    (knpy) $ git commit -m "Your detailed description of your changes."
    (knpy) $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python >= 3.6. Make sure that the 
   tests pass for all supported Python versions.

.. note:: For more on pull requests, see https://help.github.com/en/articles/about-pull-requests

Tips
----

To run a subset of tests:

.. code-block:: console

    (knpy) $ py.test tests.test_entities

