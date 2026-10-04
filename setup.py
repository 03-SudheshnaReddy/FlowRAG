"""
FlowRAG: Continual Learning for Retrieval-Augmented Generation
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh.readlines() 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="flowrag",
    version="1.0.0",
    author="FlowRAG Team",
    author_email="",
    description="A continual learning framework for Retrieval-Augmented Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/FlowRAG",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/FlowRAG/issues",
        "Documentation": "https://github.com/your-org/FlowRAG#readme",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests", "scripts", "output", "cl_datasets"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "vllm": [
            "vllm>=0.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flowrag=run_manager:main",
        ],
    },
    include_package_data=True,
    keywords=[
        "retrieval-augmented-generation",
        "continual-learning",
        "large-language-models",
        "information-retrieval",
        "nlp",
    ],
)
