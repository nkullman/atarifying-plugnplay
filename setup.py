import setuptools

setuptools.setup(
    name='atarifying',
    version='0.0.1',
    author='Nicholas Kullman',
    author_email='nicholas.kullman@etu.univ-tours.fr',
    description='Atari-like playable versions of research problems (currently just the VRPSSR)',
    packages=setuptools.find_packages(),
    entry_points={
        'console_scripts': [
            'atarify=atarifying.main:main',
        ],
    }
)
