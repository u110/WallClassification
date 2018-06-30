FROM tensorflow/tensorflow:1.5.0-py3

RUN apt-get update && \
    apt-get install -y language-pack-ja-base \
                       language-pack-ja \
                       vim \
                       tmux
ENV LANG=ja_JP.UTF-8

RUN pip install --upgrade pip \
                keras \
                flake8

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

WORKDIR "/notebooks"

# run jupyter
# CMD ["/run_jupyter.sh", "--allow-root"]

# run bash
CMD bash
