# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the requirements file into the container at /app
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Install Chainlit
RUN pip install chainlit

# Verify Chainlit installation
RUN python -m pip show chainlit
# Copy the rest of the application code into the container at /app
COPY . /app
COPY ./.chainlit ./.chainlit
COPY chainlit.md ./

# Expose port 8086 to the outside world
EXPOSE 8086

# Define environment variable for Chainlit


# Run chainlit when the container launches
# CMD ["chainlit","run",  "global-ui.py","--port","8086"]
