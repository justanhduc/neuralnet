from setuptools import setup, find_packages
import os


AUTHOR = 'DUC NGUYEN'
MAJOR = 0
MINOR = 2
MICRO = '0'
VERSION = '%d.%d.%s' % (MAJOR, MINOR, MICRO)


def write_version_py(filename='neuralnet/version.py'):
    cnt = """
#THIS FILE IS GENRERATED FROM SETUP.PY

version = '%(version)s'
author = '%(author)s'
"""
    a = open(filename, 'w')
    try:
        a.write(cnt % {'version': VERSION,
                       'author': AUTHOR})
    finally:
        a.close()


def setup_package():
    write_version_py()
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()

    setup(
        name='neuralnet',
        version=VERSION,
        description='A high-level library on top of Theano.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/justanhduc/neuralnet',
        author='Duc Nguyen',
        author_email='adnguyen@yonsei.ac.kr',
        classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: End Users/Desktop',
            'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
            'Operating System :: Microsoft :: Windows :: Windows 10',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Python :: 3.5'
        ],
        platforms=['Windows', 'Linux'],
        packages=find_packages(exclude=['examples']),
        install_requires=['theano', 'matplotlib', 'scipy', 'numpy', 'tqdm', 'visdom'],
        project_urls={
            'Bug Reports': 'https://github.com/justanhduc/neuralnet/issues',
            'Source': 'https://github.com/justanhduc/neuralnet/',
        },
    )


if __name__ == '__main__':
    setup_package()
