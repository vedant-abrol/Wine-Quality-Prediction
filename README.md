

```markdown
# CS643 - Cloud Computing Programming Assignment 2: Wine Quality Prediction

## Abstract
The objective of this final programming assignment is to create an Apache Spark MLlib application to train a machine learning model in parallel on a cluster composed of four workers and one master. This document details the step-by-step procedure for setting up the cluster, EC2 instances, and Docker images. Additionally, the document covers parallel training steps and how to run the prediction application both on a single machine without Docker and through downloading the Docker image, instantiating a container, and running it on a single machine. The code is available on GitHub, and the Docker image is available on Docker Hub. Apache Spark and Hadoop are used in this implementation.

## Training Setup

1. **On the AWS Management Console**, navigate to **Services → EC2 → Launch Instances**.
2. **Enter 5 for the number of instances**. Select the “Ubuntu Server 20.04 LTS” AMI (AMI ID: `ami-04505e74c0741db8d`).
3. **Select `t2.large` instance type** for Docker purposes.  
    - **Note**: `t2.large` instances help prevent memory-related issues. Using `t2.micro` during testing led to Java runtime environment memory errors.
4. **Create a new key pair** and name it `ProgAssgn2`. Click **Download key pair**.
5. Under **Network Settings → Security groups (Firewall)**, check **Allow SSH traffic from [Anywhere 0.0.0.0/0]**.
6. For **Configure storage**, configure it from 8 GiB to 16 GiB.
7. Launch the instance and **view all instances**. Wait for the instance status to change to Running.
8. Open a terminal and move the downloaded `.pem` file to your home directory. Set the correct permissions:
    ```
    chmod 400 ProgAssgn2.pem
    ```
9. Connect to your EC2 instance:
    ```
    ssh -i ~/ProgAssgn2.pem ubuntu@<YOUR_INSTANCE_PUBLIC_DNS>
    ```

10. Clone the GitHub repository:
    ```
    git clone https://github.com/va398/CS643-AWS-ProgAssgn-2.git
    ```

11. Copy the datasets into the `/app` directory:
    ```
    cd CS643-AWS-ProgAssgn-2
    sudo cp TrainingDataset.csv ValidationDataset.csv /app/
    ```

## Apache Spark Setup

### Bash Scripts for Apache Spark Setup

1. **Create a bash script to install, extract, and remove Apache Spark files:**
    ```
    vi automate.sh
    ```
    Insert the following lines:
    ```
    #!/bin/bash
    sudo apt-get update
    sudo apt-get install -y curl vim wget software-properties-common ssh net-tools ca-certificates
    sudo apt install -y default-jre
    sudo wget --no-verbose -O apache-spark.tgz "https://archive.apache.org/dist/spark/spark-3.2.0/spark-3.2.0-bin-hadoop2.7.tgz"
    sudo mkdir -p /opt/spark
    sudo tar -xf apache-spark.tgz -C /opt/spark --strip-components=1
    sudo rm apache-spark.tgz
    ```

2. Change permissions to execute the script:
    ```
    chmod 755 automate.sh
    ```

3. **Create the master node script**:
    ```
    vi master.sh
    ```
    Insert the following lines:
    ```
    #!/bin/bash
    export SPARK_MASTER_PORT=7077
    export SPARK_MASTER_WEBUI_PORT=8090
    export SPARK_LOG_DIR=/opt/spark/logs
    export SPARK_MASTER_LOG=/opt/spark/logs/spark-master.out
    export JAVA_HOME=/usr/bin/java
    sudo mkdir -p $SPARK_LOG_DIR
    sudo touch $SPARK_MASTER_LOG
    sudo ln -sf /dev/stdout $SPARK_MASTER_LOG
    export SPARK_MASTER_HOST=`hostname`
    cd /opt/spark/bin && 
    sudo ./spark-class org.apache.spark.deploy.master.Master --ip $SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port $SPARK_MASTER_WEBUI_PORT >> $SPARK_MASTER_LOG
    ```

4. Change permissions to execute the script:
    ```
    chmod 755 master.sh
    ```

5. **Create the slave node script**:
    ```
    vi worker<x>.sh [where <x> is 1,2,3,4]
    ```
    Insert the following lines:
    ```
    #!/bin/bash
    export SPARK_MASTER_PORT=7077
    export SPARK_MASTER_WEBUI_PORT=8080
    export SPARK_LOG_DIR=/opt/spark/logs
    export SPARK_MASTER_LOG=/opt/spark/logs/spark-master.out
    export JAVA_HOME=/usr/bin/java
    export SPARK_MASTER_PORT=7077 \
    export SPARK_MASTER_WEBUI_PORT=8080 \
    export SPARK_LOG_DIR=/opt/spark/logs \
    export SPARK_WORKER_LOG=/opt/spark/logs/spark-worker.out \
    export SPARK_WORKER_WEBUI_PORT=8080 \
    export SPARK_WORKER_PORT=7000 \
    export SPARK_MASTER="spark://ip-172-31-25-126:7077" \
    export SPARK_LOCAL_IP. "/opt/spark/bin/load-spark-env.sh"
    sudo mkdir -p $SPARK_LOG_DIR
    sudo touch $SPARK_WORKER_LOG
    sudo ln -sf /dev/stdout $SPARK_WORKER_LOG
    cd /opt/spark/bin
    sudo ./spark-class org.apache.spark.deploy.worker.Worker --webui-port $SPARK_WORKER_WEBUI_PORT $SPARK_MASTER >> $SPARK_WORKER_LOG
    ```

6. **Run the setup scripts**:
    - For the master node:
        ```
        ./master.sh
        ```
    - For each worker node:
        ```
        ./worker<x>.sh
        ```

## Training

1. **Install Python dependencies**:
    ```
    sudo apt install python3-pip
    pip install numpy
    pip install pandas
    pip install quinn
    pip install pyspark
    pip install findspark
    ```

2. **Run the training script**:
    ```
    python3 /home/ubuntu/CS643-AWS-ProgAssgn-2/Training.py
    ```

3. The output results from the script:
    - **F1 Score for LogisticRegression Model**: `0.5729445029855991`
    - **F1 Score for RandomForestClassifier Model**: `0.5035506965944272`

4. The **LogisticRegression Model** has a higher score and is used in the prediction application.

---

## Prediction without Docker

1. **Install and configure Java**:
    ```
    export JAVA_HOME=/usr/bin/java
    ```

2. **Install Anaconda**:
    ```
    cd /tmp
    curl -O https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    bash Anaconda3-2020.11-Linux-x86_64.sh
    ```

3. **Modify `.bashrc`** to add Spark configurations:
    ```
    function snotebook ()
    {
    SPARK_PATH=~/opt/spark/spark-3.2.0-bin-hadoop2.7
    export PYSPARK_DRIVER_PYTHON="jupyter"
    export PYSPARK_DRIVER_PYTHON_OPTS="notebook"
    export PYSPARK_PYTHON=python3
    $SPARK_PATH/bin/pyspark --master local[2]
    }
    ```

4. **Start Jupyter Notebook**:
    ```
    jupyter notebook password
    jupyter notebook --no-browser
    ```

5. **SSH tunnel to open Jupyter on your browser**:
    ```
    ssh -i "ProgAssgn2.pem" -N -f -L localhost:8888:localhost:8888 ubuntu@<YOUR_INSTANCE_PUBLIC_DNS>
    ```

6. **Navigate to localhost:8888** in your browser and enter the Jupyter password.

---

## Prediction with Docker

1. **Initialize Docker on EC2**:
    ```
    sudo usermod -aG docker $USER
    ```

2. **Login to Docker Hub**:
    ```
    docker login
    ```

3. **Navigate to the Dockerfile directory**:
    ```
    cd /home/ubuntu/CS643-AWS-ProgAssgn-2/
    ```

4. **Build Docker image**:
    ```
    docker build -t va398/aws-cs643-progassgn-2 .
    ```

5. **Verify the Docker image has been created**:
    ```
    docker images
    ```

6. **Run the Docker container**:
    ```
    docker run -v /app/:/data va398/aws-cs643-progassgn-2:latest
    ```

7. **After running the predictions with Docker**:
    - **F1 Score for our Model**: `0.562631807944308`

8. **Push the image to Docker Hub**:
    ```
    docker push va398/aws-cs643-progassgn-2
    ```

---

## Conclusion
This guide provides step-by-step instructions to set up the EC2 instances, configure Apache Spark, train a machine learning model using Spark, and run predictions both with and without Docker. All results, including Docker images, have been successfully deployed and tested.

---

