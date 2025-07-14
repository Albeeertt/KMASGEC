from setuptools import setup, find_packages

with open("requeriments.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="kmasgec",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
      package_data={                    
        "kmasgec": ["models/*.pt"],
    },
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'KMAsgec=kmasgec.cli:main',
        ],
    },
)
