FROM nvidia/cuda:11.3.0-devel-ubuntu20.04
RUN export DEBIAN_FRONTEND=noninteractive && \
    export UBUNTU_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 &&\
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu focal main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt update && \
    apt install tzdata -y && \
    apt install keyboard-configuration -y && \
    apt install python3-pip -y && \
    apt install ros-noetic-desktop -y && \
    apt install ros-noetic-novatel-oem7-msgs -y && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
RUN pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113 && pip install timm
RUN apt install git-lfs -y && \
    apt install libpcl-dev -y && \
    pip install scikit-learn pandas opencv-python laspy && \
    cd /root && git clone https://github.com/ebhrz/HDmap_dependencies.git && cd HDmap_dependencies &&\
    cat pclpy.tar.gz* | tar zx && mv pclpy /usr/local/lib/python3.8/dist-packages/ && \
    tar xzf ops.tar.gz && mv MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg /usr/local/lib/python3.8/dist-packages/ && \
    echo './MultiScaleDeformableAttention-1.0-py3.8-linux-x86_64.egg' > /usr/local/lib/python3.8/dist-packages/easy-install.pth && \
    cp so/* /usr/lib/x86_64-linux-gnu/
RUN pip install detectron2==0.6 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html && \
    pip install -U openmim && \
    mim install mmcv-full && \
    pip install mmsegmentation && \
    pip install pymap3d