# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:56:52 2024

@author: stevf
"""

import os

# Replace 'your_library_name' with the name of your library
library_name = 'analytica'

# Define the directory structure and files to be created
structure = {
    f'{library_name}/': [
        '__init__.py',
        'core.py',
        'transformations.py',
        'utils.py',
    ],
    'tests/': [
        '__init__.py',
        f'test_core.py',
        f'test_transformations.py',
    ],
    'examples/': [
        'basic_usage.py',
    ],
    'docs/': [
        'index.md',
    ],
}

# Files outside the specific directories
root_files = [
    'setup.py',
    'README.md',
    'LICENSE',
    '.gitignore',
]

def create_files(base_path, file_list):
    """Create files with basic content."""
    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        with open(file_path, 'w') as f:
            if file_name == 'README.md':
                f.write(f'# {library_name}\n\nA brief description of the library.\n')
            elif file_name == 'LICENSE':
                f.write('MIT License\n\nCopyright (c) [year] [fullname]\n\nPermission is hereby granted...')
            elif file_name == '.gitignore':
                f.write('.DS_Store\n__pycache__/\n*.pyc\n')
            elif file_name == 'setup.py':
                f.write('from setuptools import setup, find_packages\n\nsetup(\n    name="'+library_name+'",\n    version="0.1",\n    packages=find_packages(),\n)')
            elif file_name.endswith('.py'):
                f.write('# Placeholder for ' + file_name)
            elif file_name == 'index.md':
                f.write(f'# Welcome to {library_name}\n\nThis is the starting page of the documentation.')

def create_structure(base_path, structure_dict):
    """Recursively create the directory structure and files."""
    for directory, files in structure_dict.items():
        dir_path = os.path.join(base_path, directory)
        os.makedirs(dir_path, exist_ok=True)
        create_files(dir_path, files)

# Create the library structure
create_structure('.', structure)
# Create files in the root directory
create_files('.', root_files)

print(f'Library structure for "{library_name}" created successfully.')
