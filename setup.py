from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='Presto Chango',
    version='1.0.1',
    author='Prayag Savsani',
    author_email='prayag.s@ahduni.edu.in',
    description='Music identification through audio fingerprinting',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/yashrajkakkad/presto-chango',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Sound/Audio :: Analysis'
    ],
    install_requires=[
        'Click', 'scipy', 'numpy', 'scikit-image', 'matplotlib', 'pydub', 'pyaudio'
    ],
    entry_points={
        'console_scripts': [
            'presto-chango = presto_chango:cli'
        ]
    }
)
