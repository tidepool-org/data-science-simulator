from setuptools import setup

setup(
    name='Tidepool Data Science Simulator',
    version="0.1.0",
    author="Cameron Summers",
    author_email="cameron@tidepool.org",
    package_dir={'tidepool_data_science_simulator': 'src'},
    packages=[
        'tidepool_data_science_simulator',
        'tidepool_data_science_models.evaluation',
        'tidepool_data_science_models.legacy',
        'tidepool_data_science_models.makedata',
        'tidepool_data_science_models.models',
        'tidepool_data_science_models.vizualization',
    ],
    long_description=open('README.md').read(),
    python_requires='>=3.6',
    license="BSD-2",
)
