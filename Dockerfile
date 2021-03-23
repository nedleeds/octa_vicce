FROM tensorflow/tensorflow:2.2.2-gpu-py3-jupyter

RUN apt-get update &&\
    install sudo \
    build-essential \
    cmake \
    git \
    wget \
    libatlas-base-dev \
    libboost-all-dev \
    libgflags-dev \
    libgoogle-glog-dev \
    libhdf5-serial-dev \
    libleveldb-dev \
    liblmdb-dev \
    libopencv-dev \
    libprotobuf-dev \
    libsnappy-dev \
    protobuf-compiler \
    python-dev \
    python-pip \
    libgl1-mesa-glx &&\
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip &&\
    install opencv-python\
    scipy \
    numba \
    numpy \
    ipython \
    keras \
    opencv-python \
    pandas \
    Image \
    matplotlib \
    sklearn \
    gpustat \
    -q tensorflow-probability\
    -q imageio \
    -q git+https://github.com/tensorflow/docs

# setting ZSH
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
-a 'CASE_SENSITIVE="true"'
RUN echo "alias python=\"python3\"" >> ~/.zshrc
RUN echo "alias code=\"/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code\"" >> ~/.zshrc





# FROM tensorflow/tensorflow:latest-gpu

# ARG DEBIAN_FRONTEND=noninteractive
# RUN apt-get update 
# RUN apt-get install -y --no-install-recommends 
# RUN apt-get install \
#         sudo \
#         build-essential \
#         cmake \
#         git \
#         wget 
#         # libatlas-base-dev \
#         # libboost-all-dev \
#         # libgflags-dev \
#         # libgoogle-glog-dev \
#         # libhdf5-serial-dev \
#         # libleveldb-dev \
#         # liblmdb-dev \
#         # libopencv-dev \
#         # libprotobuf-dev \
#         # protobuf-compiler \
#         # python-dev \
#         # python-pip \
#         # libgl1-mesa-glx &&\
#     # rm -rf /var/lib/apt/lists/*

# RUN pip install --upgrade pip && install zsh \
#         scipy \
#         numba \
#         numpy \
#         ipython \
#         keras \
#         opencv-python \
#         pandas \
#         Image \
#         matplotlib \
#         sklearn 
# # gpustat => nvidia-smi daemon => gpustat -i


# # RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" \
# #     echo "alias python=\"python3\"" >> ~/.zshrc \
# #     echo "alias code=\"/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code\"">> ~/.zshrc \
# #     chsh -s \\`which zsh\\`

# RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
#     -t https://github.com/ohmyzsh/ohmyzsh/blob/master/themes/cloud.zsh-theme \
#     -p git \
#     -p ssh-agent \
#     -p https://github.com/zsh-users/zsh-autosuggestions \
#     -p https://github.com/zsh-users/zsh-completions \
#     echo "alias python=\"python3\"" >> ~/.zshrc \
#     echo "alias code=\"/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code\"">> ~/.zshrc \
#     chsh -s \\`which zsh\\`

# CMD ["cd", ".."] \
#     ["cd", "root"]
