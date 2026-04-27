# IP102 Pest Identification Web App

Deep learning pest classification project using a ResNet-50 model trained on IP102 (102 classes), with a Django web app for drag-and-drop image inference.

## Highlights
- 102-class pest classification pipeline (IP102)
- Saved model loading for web inference
- Browser UI for image upload and Top-K predictions
- Organized project layout for training, artifacts, and deployment

## Reported Metrics
- Top-1 Accuracy: 73.60%
- Top-3 Accuracy: 88.80%
- Top-5 Accuracy: 92.40%

## Project Structure
- notebooks/: training and experimentation notebook
- data/: dataset folder and source zip
- artifacts/: generated checkpoints, reports, and model outputs
- webapp/: Django application for inference

## Tech Stack
Python, PyTorch, Torchvision, Albumentations, OpenCV, NumPy, scikit-learn, Django, HTML, CSS, JavaScript

## Quick Start (Web App)
1. Install dependencies:
   - pip install -r requirements.txt
2. Move into web app folder:
   - cd webapp
3. Apply migrations:
   - python manage.py migrate
4. Start server:
   - python manage.py runserver 127.0.0.1:8000
5. Open in browser:
   - http://127.0.0.1:8000/

## Model Artifacts Policy
By default, large artifacts are ignored in .gitignore (for cleaner and faster Git history), including:
- data/dataset/
- data/IP102-Dataset.zip
- artifacts/

This is recommended for most repos. If you want to share model weights publicly, prefer one of these:
- GitHub Releases (attach selected model files)
- External storage (Google Drive/Hugging Face/Kaggle) with links in this README

## Notes
- Ensure your model files exist under artifacts/ before running inference.
- For deployment, keep secrets/config out of source control.
