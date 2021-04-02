FROM tensorflow/tensorflow:2.2.2-gpu-py3
WORKDIR /root/Share
ENV DEBIAN_FRONTEND noninteractive

RUN ["apt-get", "update"]
RUN ["apt-get", "install", "-y", "--no-install-recommends", "apt-utils"]
RUN ["apt-get", "install", "-y", "zsh"]
RUN ["apt-get", "install", "libgl1-mesa-glx", "-y"]
RUN ["apt-get", "install", "libncurses5-dev", "-y"]
RUN ["apt-get", "install", "libncursesw5-dev", "-y"]
RUN ["apt-get", "install", "git", "-y"]
RUN ["apt-get", "install", "make"]
RUN ["apt-get", "install", "build-essential"]
RUN ["apt-get", "install", "wget"]
RUN ["apt-get", "install", "vim", "-y"]
RUN ["apt-get", "install", "libglib2.0-0", "-y"]

RUN pip install --upgrade pip \
    pip install opencv-python \
    scipy \
    numba \
    numpy==1.18.1 \
    ipython \
    ipykernel \
    pandas \
    Image \
    matplotlib \
    sklearn \
    gpustat \
    keras \
    tensorflow-probability==0.10.1

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.1.1/zsh-in-docker.sh)" -- \
    -t cloud \
    -p git \
    -p ssh-agent \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/zsh-users/zsh-history-substring-search \
    -p https://github.com/zsh-users/zsh-syntax-highlighting \
    -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down'

RUN chsh -s /bin/zsh    
RUN PATH="$PATH:~/bin/zsh:/usr/bin/zsh:/bin/zsh/:/zsh:/Applications/Visual Studio Code.app/Contents/Resources/app/bin"
RUN echo "ZSH_THEME_CLOUD_PREFIX='🌤️'" >> ~/.zshrc
RUN echo "ZSH_THEME_GIT_PROMPT_DIRTY=\"%{\$fg[green]%}] %{\$fg[yellow]%}🔥️%{\$reset_color%}\"" >> ~/.zshrc
RUN echo "ZSH_THEME_GIT_PROMPT_CLEAN=\"%{\$fg[green]%}] 🚀️\"" >> ~/.zshrc
RUN echo "PROMPT+=$'\n➤➤ '" >> ~/.zshrc
RUN echo "alias python=\"python3\"" >> ~/.zshrc

CMD ["/bin/zsh"]