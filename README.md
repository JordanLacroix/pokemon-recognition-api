# Pokemon Recognition API

This is a Flask API for recognizing Pokemon using a deep learning model trained on a dataset of 150 Pokemon. 

## Getting started

1. Clone this repo
2. Install the dependencies: `pip install -r requirements.txt`
3. Start the server: `python app.py`

## Usage

Send a POST request to the `/predict` endpoint with an image file in the `file` field of the form data. The response will contain the predicted Pokemon.

## Retraining the model

To add a new Pokemon to the recognition model, send a POST request to the `/retrain` endpoint with an image file in the `file` field of the form data and the correct class label in the `class` field of the form data.

## Security

This API includes rate limiting to prevent abuse. Requests are limited to 10 per minute per IP address. 

## Optimization for Production

This API can be optimized for production by using a production-ready web server such as Gunicorn and deploying on a cloud platform such as Azure. 

