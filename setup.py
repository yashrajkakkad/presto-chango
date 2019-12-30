from setuptools import setup, find_packages

setup(
    name='Presto Chango',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'Click', 'scipy', 'numpy', 'scikit-image', 'matplotlib', 'pydub', 'pyaudio'
    ],
    entry_points={
        'console_scripts': [
            'presto-chango = presto_chango:cli'
        ]
    }
)
