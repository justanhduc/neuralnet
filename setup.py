from setuptools import setup, find_packages
import codecs
import os


here = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    with codecs.open(os.path.join(here, *parts), "rb", "utf-8") as f:
        return f.read()


with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='neuralnet',
    version='0.0.1',
    description='A high-level library on top of Theano.',
    long_description=read('readme.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/justanhduc/neuralnet',
    author='Duc Nguyen',
    author_email='adnguyen@yonsei.ac.kr',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Practitioners',
        'License :: OSI Approved :: Mozilla Public License 2.0',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=['theano==0.9.0', 'matplotlib', 'scipy', 'numpy'],
    project_urls={
        'Bug Reports': 'https://github.com/justanhduc/neuralnet/issues',
        'Source': 'https://github.com/justanhduc/neuralnet/',
    },
)