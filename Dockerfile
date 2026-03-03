# ============================================================
# Dockerfile for Physics-Constrained Fake BS Detection
# ns-3.39 + Python 3 + numpy + pandas + scipy
# ============================================================
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NS3_VERSION=3.39
ENV NS3_DIR=/opt/ns-allinone-3.39/ns-3.39

# ---- System dependencies ----
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    tar \
    git \
    libsqlite3-dev \
    libxml2-dev \
    libgtk-3-dev \
    gsl-bin \
    libgsl-dev \
    libgslcblas0 \
    ccache \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# ---- Python packages ----
RUN pip3 install --no-cache-dir numpy pandas scipy

# ---- Download and build ns-3.39 ----
WORKDIR /opt
RUN wget -q https://www.nsnam.org/releases/ns-allinone-3.39.tar.bz2 \
    && tar xjf ns-allinone-3.39.tar.bz2 \
    && rm ns-allinone-3.39.tar.bz2

WORKDIR /opt/ns-allinone-3.39/ns-3.39

# Configure ns-3 with LTE and examples disabled (faster build)
RUN ./ns3 configure \
    --enable-modules=lte,internet,mobility,buildings,point-to-point,applications \
    --disable-examples \
    --disable-tests \
    --build-profile=optimized

# Build the configured modules (no scratch yet)
RUN ./ns3 build

# ---- Runtime working directory ----
WORKDIR /opt/ns-allinone-3.39/ns-3.39

# Default command — overridden by docker-compose
CMD ["/bin/bash"]
