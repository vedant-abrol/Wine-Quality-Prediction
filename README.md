# Wine Quality Prediction

## Overview
This project aims to develop a machine learning model to predict the quality of wine using Apache Spark MLlib. The model is trained in parallel on a cluster of AWS EC2 instances and deployed using Docker for efficient execution.

## Table of Contents
- [Project Overview](#overview)
- [Setup and Installation](#setup-and-installation)
  - [AWS EC2 Setup](#aws-ec2-setup)
  - [Apache Spark Setup](#apache-spark-setup)
- [Training the Model](#training-the-model)
- [Prediction](#prediction)
  - [Without Docker](#prediction-without-docker)
  - [With Docker](#prediction-with-docker)
- [Results](#results)
- [Conclusion](#conclusion)

---

## Setup and Installation

### AWS EC2 Setup
1. **Launch EC2 Instances**
   - Go to **AWS Management Console** > **EC2** > **Launch Instances**
   - Set **Number of instances** to **5**
   - Choose **Ubuntu Server 20.04 LTS** (AMI ID: `ami-04505e74c0741db8d`)
   - Select `t2.large` instance type
   - Configure storage: Increase from **8 GiB** to **16 GiB**
   - Allow **SSH traffic from Anywhere (0.0.0.0/0)`** in **Security Groups**
   - Create and download the key pair (`ProgAssgn2.pem`)
   - Launch the instances and wait until they are **Running**

2. **Connect to the EC2 Instance**
   ```sh
   chmod 400 ProgAssgn2.pem
   ssh -i ProgAssgn2.pem ubuntu@<INSTANCE_PUBLIC_DNS>
   ```

3. **Clone the GitHub Repository**
   ```sh
   git clone https://github.com/vedant-abrol/Wine-Quality-Prediction.git
   ```

4. **Move the datasets into the application directory**
   ```sh
   cd Wine-Quality-Prediction
   sudo cp TrainingDataset.csv ValidationDataset.csv /app/
   ```

### Apache Spark Setup
1. **Install Dependencies**
   ```sh
   sudo apt update
   sudo apt install -y default-jre curl wget ssh net-tools
   ```

2. **Download and Install Apache Spark**
   ```sh
   wget -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop2.7.tgz"
   sudo mkdir -p /opt/spark
   sudo tar -xf apache-spark.tgz -C /opt/spark --strip-components=1
   rm apache-spark.tgz
   ```

3. **Setup Spark Master Node**
   ```sh
   echo "export SPARK_MASTER_PORT=7077" >> ~/.bashrc
   echo "export SPARK_MASTER_WEBUI_PORT=8090" >> ~/.bashrc
   source ~/.bashrc
   /opt/spark/bin/spark-class org.apache.spark.deploy.master.Master &
   ```

4. **Setup Worker Nodes**
   ```sh
   /opt/spark/bin/spark-class org.apache.spark.deploy.worker.Worker spark://<MASTER_NODE_IP>:7077 &
   ```

---

## Training the Model

1. **Install Python Dependencies**
   ```sh
   sudo apt install python3-pip
   pip install numpy pandas pyspark findspark quinn
   ```

2. **Run the Training Script**
   ```sh
   python3 Training.py
   ```

3. **Model Performance Metrics**
   - **Logistic Regression F1 Score**: `0.5729`
   - **Random Forest Classifier F1 Score**: `0.5035`

   The **Logistic Regression** model performs better and is used for predictions.

---

## Prediction

### Prediction Without Docker
1. **Configure Java Environment**
   ```sh
   export JAVA_HOME=/usr/bin/java
   ```
2. **Run Prediction Script**
   ```sh
   python3 Predictions.py
   ```

### Prediction With Docker

1. **Install and Configure Docker**
   ```sh
   sudo usermod -aG docker $USER
   docker login
   ```

2. **Build and Run Docker Container**
   ```sh
   cd Wine-Quality-Prediction
   docker build -t wine-quality-predictor .
   docker run -v /app/:/data wine-quality-predictor
   ```

3. **Push Docker Image to Docker Hub**
   ```sh
   docker tag wine-quality-predictor <DOCKERHUB_USERNAME>/wine-quality-predictor
   docker push <DOCKERHUB_USERNAME>/wine-quality-predictor
   ```

---

## Results
- **Logistic Regression Model F1 Score (Docker Execution)**: `0.5626`
- The Dockerized model achieves similar accuracy as the manually executed script.

---

## Conclusion
This project successfully demonstrates the end-to-end process of setting up an Apache Spark cluster on AWS EC2, training a machine learning model, and deploying it with Docker for scalable wine quality prediction.

For any questions or contributions, feel free to open an issue or fork the repository!

---

