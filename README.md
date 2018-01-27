## Dependencies

This code depends on numpy and scipy and works only with Python 3.

For the functions written in Rust to be available, you need Rust on your system.
See <https://www.rust-lang.org/en-US/install.html>.

## Setting up

Clone the repository into your favorite location. To set up the development
environment, run `python setup.py develop --user`.

Note: If you changed any Rust code, for that to take effect, you need to rerun
`setup.py`.

## Usage

Please refer to the docstrings of individual functions and classes.

All hamiltonian definitions go into the "hamiltonians" folder and all scripts,
big and small, go into the "scripts" folder. Only reusable Python code goes into
"spinsys".

All Rust code goes into "rust".

## License

All code in this repository is released under the BSD 3-clause license. For
details please refer to LICENSE.txt.
