from setuptools import setup

setup(
    name='looking_glass',
    version='0.1.0',
    packages=['looking_glass'],
    author='',
    author_email='',
    description='Interpretable clustering',
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'shap', 'interpret'
    ],
    include_package_data=True,
    url='',
)
