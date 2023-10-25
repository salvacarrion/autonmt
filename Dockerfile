FROM pytorch/pytorch:latest

# Update and install required tools
RUN apt-get update &&  \
    apt-get install -y python3-pip &&  \
    apt-get install -y vim && \
    apt-get install -y htop && \
    apt-get install -y tmux

# Set the working directory in the docker container
WORKDIR /app

# Create a directories
RUN mkdir /app/autonmt
RUN mkdir /app/data


# Copy the content of the local src directory to the working directory
COPY . /app/autonmt

# Install the python packages
RUN pip install --upgrade pip
RUN pip install -e autonmt

# This command will run when the container starts.
CMD ["tail", "-f", "/dev/null"]
