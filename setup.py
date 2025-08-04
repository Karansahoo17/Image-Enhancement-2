from setuptools import setup, find_packages

setup(
    name="ai-image-pipeline", 
    version="1.0.0",
    description="AI Image Processing Pipeline with Sketch-to-Image, Face Editing, and Image Restoration",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        line.strip() 
        for line in open("requirements.txt").readlines()
        if not line.startswith("#") and line.strip()
    ],
    entry_points={
        "console_scripts": [
            "ai-pipeline=main:main",
        ],
    },
)
