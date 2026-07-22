from setuptools import setup

setup(
    name='Tidepool Data Science Simulator',
    version="0.1.0",
    author="Cameron Summers",
    author_email="cameron@tidepool.org",
    package_dir={'tidepool_data_science_simulator': 'tidepool_data_science_simulator'},
    packages=[
        'tidepool_data_science_simulator',
        'tidepool_data_science_simulator.evaluation',
        'tidepool_data_science_simulator.legacy',
        'tidepool_data_science_simulator.makedata',
        'tidepool_data_science_simulator.models',
        'tidepool_data_science_simulator.post_processing',
        'tidepool_data_science_simulator.visualization',
        'tidepool_data_science_simulator.projects',
        'tidepool_data_science_simulator.projects.risk',
        'tidepool_data_science_simulator.validation',
    ],
    long_description=open('README.md').read(),
    python_requires='>=3.6',
    license="BSD-2",
)
