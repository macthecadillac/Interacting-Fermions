from setuptools import setup, find_packages
from subprocess import Popen
import shutil
import os
import sys


# key = module name (the exact name of the cargo project)
# val = destination of the generated shared object
# header files must have the same name as the module and be present in the root
# of the Rust project dir
rust_modules = {
    "triangular_lattice_ext": "models",
}


def build_rust_binary(lib, dest_dir):
    os.chdir("rust/{}".format(lib))
    p = Popen(["cargo", "build", "--release"])
    p.wait()
    if p.returncode != 0:
        sys.exit("Cargo build failure. Abort.")
    os.chdir("../..")


if shutil.which("cargo"):
    for lib, dest_dir in rust_modules.items():
        build_rust_binary(lib, dest_dir)
else:
    print("Cargo not found in path. Continue without building Rust modules.")
    print("Note: The Rust-based functions will not be available.")


setup(
    name="spinsys",
    packages=find_packages(),
    install_requires=['numpy>=1.8', 'scipy>=0.13', 'cffi>=1.11'],
    include_package_data=True,
)
