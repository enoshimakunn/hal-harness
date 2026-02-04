FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda (supports both x86_64 and aarch64 builders)
RUN ARCH="$(uname -m)" \
    && if [ "$ARCH" = "x86_64" ]; then CONDA_INSTALLER=Miniconda3-latest-Linux-x86_64.sh; \
       elif [ "$ARCH" = "aarch64" ]; then CONDA_INSTALLER=Miniconda3-latest-Linux-aarch64.sh; \
       else echo "Unsupported architecture: $ARCH" && exit 1; fi \
    && wget "https://repo.anaconda.com/miniconda/${CONDA_INSTALLER}" -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate base" >> ~/.bashrc \
    && /opt/conda/bin/conda clean -afy

# Add conda to PATH
ENV PATH /opt/conda/bin:$PATH

# Install weave/wandb combo and pin gql < 4
RUN pip install "wandb>=0.17.2,<0.19.0" weave==0.51.41 "gql<4"

# Set working directory
WORKDIR /workspace

# We don't use a non-root user because we need to write files
# that can be read by the host system after the container exits

# Default entry point
CMD ["bash"] 