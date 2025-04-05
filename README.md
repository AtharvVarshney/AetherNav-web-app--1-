# Wearable AI Assistant for Visually Impaired

A comprehensive AI-powered assistant designed to help visually impaired individuals navigate both indoor and outdoor environments while providing various recognition capabilities.

## Features

### 1. Text Recognition & Reading
- Real-time text detection and reading from images and surroundings
- Converts detected text to speech for immediate audio feedback
- Supports multiple languages and various text formats

### 2. Object Detection & Recognition
- Real-time object detection and identification
- Provides audio feedback about detected objects
- Helps users identify everyday items, obstacles, and potential hazards
- Distance estimation to detected objects

### 3. Face Detection & Recognition
- Detects faces in real-time
- Recognizes registered faces of family members, friends, and acquaintances
- Provides audio feedback about who is present in front of the user
- Helps in social interactions by identifying familiar people

### 4. Navigation Systems

#### Outdoor GPS Navigation
- Turn-by-turn voice navigation
- Real-time location tracking
- Safe route suggestions
- Distance and estimated time calculations
- Voice-activated destination input

#### Indoor Navigation
- Indoor positioning system
- Room and area identification
- Obstacle avoidance
- Doorway and exit detection
- Safe path planning within buildings

## Requirements

- Python 3.x
- Required libraries (install via pip):
  ```bash
  pip install opencv-python
  pip install numpy
  pip install tensorflow
  pip install gtts
  pip install playsound
  pip install face_recognition