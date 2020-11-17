from setuptools import setup

setup(
    name='interpret_clusters',
    version='0.1.0',
    packages=['interpret_clusters'],
    author='',
    author_email='',
    description='Interpretable clustering',
    install_requires=[
        'numpy', 'pandas', 'scikit-learn', 'shap', 'interpret'
    ],
    include_package_data=True,
    url='',
)
