#
#  Copyright (c) 2016-2022 Deephaven Data Labs and Patent Pending
#
import os
import pathlib

# Note: pkg_resources is deprecated https://setuptools.pypa.io/en/latest/pkg_resources.html, and it is suggested
# to use an external library `packaging`. From the context of building a wheel though, we'd prefer to not have to
# install extra dependencies, at least until we can more properly manage the build environment (pyproject.toml).
# TODO(deephaven-core#2233): upgrade setup.py to pyproject.toml
from pkg_resources import parse_version
from setuptools import find_namespace_packages, setup

def _get_readme() -> str:
    # The directory containing this file
    HERE = pathlib.Path(__file__).parent
    # The text of the README file
    return (HERE / "README_PyPi.md").read_text(encoding="utf-8")

def _normalize_version(java_version) -> str:
    partitions = java_version.partition("-")
    regular_version = partitions[0]
    local_segment = partitions[2]
    python_version = f"{regular_version}+{local_segment}" if local_segment else regular_version
    return str(parse_version(python_version))

def _compute_version():
    return _normalize_version(os.environ['DEEPHAVEN_VERSION'])

_version = _compute_version()

setup(
    name='deephaven-server',
    version=_version,
    description='Deephaven Embedded Server Python Package',
    long_description=_get_readme(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=("tests")),
    package_data={'deephaven_server': ['jars/*', 'py.typed']},
    url='https://deephaven.io/',
    author='Deephaven Data Labs',
    author_email='python@deephaven.io',
    license='Deephaven Community License',
    test_loader='unittest:TestLoader',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Other/Proprietary License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='Deephaven Development',
    python_requires='>=3.8',
    install_requires=[
        'jpy>=0.14.0',
        "java-utilities",
        f"deephaven-core[autocomplete]=={_version}",
    ]
)
