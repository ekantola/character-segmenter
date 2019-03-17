import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='character-segmenter',
    version='0.1',
    author='Eemeli Kantola',
    author_email='eemeli.kantola@iki.fi',
    description='Segment and separate characters in a bitmap image',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ekantola/character-segmenter.git',
    packages=setuptools.find_packages(),
    py_modules=['segmenter', 'display_segments', 'sample_image'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
