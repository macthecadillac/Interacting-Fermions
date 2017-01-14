# Dependencies
The following packages will need to be installed: numpy, scipy, msgpack-python 

Numpy and Scipy should come pre-installed if you are using the Anaconda
distribution. Msgpack will need to be installed separately via
```shell
pip install --user msgpack-python
```

# Installation
Copy and paste the following in the terminal
```shell
git clone https://github.com/1119group/spinsys
```
You will now find the folder in your home directory. To make the library available 
in the prompt, link the folder to the local Python path. On Linux, the folder
should be linked to 
~/.local/lib/python3.x/site-packages/ where x is the currrent default Python
version on your system. Once that is done, the library is available system-wide
(for your user) and you can import it the usual way.

# Usage
The library can be imported with
```python
import spinsys
```
It is prefereable that modules within the spinsys folder be left unmodified unless the
function written is general enough to be sorted into one of the
sub-modules/sub-folders. Temporary scripts for plotting and simulation should
be put somewhere else to keep the library/repository clean and well-organized.

# Commits
Everybody that committed new code needs to own their code--they need to fix bugs
in their code/caused by their code when problems arise, and they need to make
their code as readable/usable as possible by putting in a reasonable amount
of documentation (docstrings, comments).
