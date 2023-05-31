import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

DOCS_REQUIRE = [
    "sphinx~=6.2.1",
    "sphinx-autodoc-typehints~=1.23.0",
    "sphinx-rtd-theme~=1.2.0",
]

setuptools.setup(
    name='itmobotics_sim',
    version='0.0.6',
    author='TexnoMann',
    author_email='texnoman@itmo.ru',
    description='Package with pybullet robots simulation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ITMORobotics/itmobotics_sim",
    project_urls={
        "Bug Tracker": "https://github.com/ITMORobotics/itmobotics_sim/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
    download_url = 'https://github.com/ITMORobotics/itmobotics_sim/archive/refs/tags/v0.0.6.tar.gz',
    include_package_data=True,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={"": ["README.md", "LICENSE.txt"]},
    install_requires=[
        "jsonschema",
        "numpy >=1.20.0",
        "scipy >= 1.7.0",
        "pybullet >= 3.1.9",
        "sympy",
        "spatialmath-python>=0.11",
        "urdf-parser-py"
   ],
   extras_require={
        "docs": DOCS_REQUIRE,
    },
)
