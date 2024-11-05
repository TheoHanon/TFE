from setuptools import setup, find_packages

setup(
    name='spherical_siren',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy'  # Assuming numpy is required, as it's commonly used in ML models
    ],
    extras_require={
        'dev': [
            'pytest',  # For running tests
            'flake8'   # For code style checking
        ]
    },
    # author='Theo Hanon',
    # author_email='your.email@example.com',
    # description='A package implementing Spherical Harmonics Embedding and Spherical SIREN models.',
    # long_description=open('README.md').read(),
    # long_description_content_type='text/markdown',
    # # url='https://github.com/yourusername/spherical_siren',
    # classifiers=[
    #     'Programming Language :: Python :: 3',
    #     'License :: OSI Approved :: MIT License',
    #     'Operating System :: OS Independent',
    # ],
    python_requires='>=3.6',
)