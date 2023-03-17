from setuptools import setup, find_packages

setup(
    name='depth_c2rp',
    version='0.4.1',
    description='Depth Assisted Camera To Robot Pose Estimation',
    url='https://github.com/sithu31296/semantic-segmentation',
    author='Sithu Aung',
    author_email='sithu31296@gmail.com',
    license='MIT',
    packages=find_packages(include=['depth_c2rp']),
    install_requires=[
        'tqdm',
        'tabulate',
        'numpy',
        'scipy',
        'matplotlib',
        'tensorboard',
        'fvcore',
        'einops',
        'rich',
    ]
)