from setuptools import setup

setup(
    name='interpret_clusters',
    version='0.1.0',
    packages=['interpret_clusters'],
    author='',
    author_email='',
    description='Try to interpret unsupervised learning models by training a series of one-vs-all classifiers.',
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'interpret'
    ],
    include_package_data=True,
    url='https://github.com/CybercentreCanada/interpret-clusters',
)
