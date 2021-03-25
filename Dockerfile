FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

WORKDIR /root

RUN pip install opencv-python \
    scipy \
    numba \
    numpy \
    ipython \
    pandas \
    Image \
    matplotlib \
    sklearn \
    gpustat \
    keras \
    ipykernel \
    tensorflow_probability==0.10.1

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN ["apt-get", "install", "libgl1-mesa-glx", "-y"]
RUN ["apt-get", "install", "git", "-y"]
RUN ["apt-get", "install", "wget"]
RUN ["apt-get", "install", "build-essential"]
RUN ["apt-get", "install", "apt-get install libglib2.0-0", "-y"]
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t https://github.com/denysdovhan/spaceship-prompt \
    -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
    -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions

RUN echo "alias python=\"python3\"" >> ~/.zshrc
RUN echo "alias code=\"/Applications/Visual\ Studio\ Code.app/Contents/Resources/app/bin/code\"" >> ~/.zshrc \
    PATH="$PATH:~/bin/zsh:/usr/bin/zsh:/bin/zsh/:/zsh"

RUN chsh -s /bin/zsh
CMD ["/bin/zsh"]


