FROM fokkodriesprong/docker-pyspark

# Set the working directory inside the container
WORKDIR /app

# Copy your project files into the container's working directory
COPY . /app

# Install any dependencies from requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Command to run when the container starts
CMD ["python", "Predictions.py"]


LABEL maintainer="va398" description="CS643 Assignment 2 - Wine Quality Prediction"
