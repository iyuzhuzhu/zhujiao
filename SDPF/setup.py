from setuptools import setup, find_packages

setup(
    name='SDPF',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    entry_points={
        'console_scripts': [
            'masterproc = MasterProc.main:main',
            'rms = rms.main:main',
            'trend = trend.main:main',
            'ai = ai.main:main'
        ]
    },
    author='J-TEXT 103',
    author_email='zhenwei@hust.edu.cn',
    description='Master Process module',
    license='MIT',
)