#!/bin/bash


if ! command -v python3.7 &> /dev/null
then
    echo "Python 3.7 could not be found. Please install Python 3.7 before proceeding."
    exit
fi


if [ ! -d "venv" ]; then

  echo "Creating virtual environment with Python 3.7..."
  python3.7 -m venv venv
else
  echo "Virtual environment already exists."
fi


echo "Activating virtual environment..."
source venv/bin/activate


echo "Upgrading pip..."
pip install --upgrade pip


echo "Installing dependencies..."
pip install -r requirements.txt


echo "Starting the training with imageDetector.py..."
python imageDetector.py


deactivate

echo "Training complete. Virtual environment deactivated."
