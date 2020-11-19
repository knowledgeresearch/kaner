FROM nvidia/cuda:10.1-base-ubuntu18.04
LABEL maintainer="Knowledge Research"
ENV LANG en_US.UTF-8
# If the official source of Ubuntu is slow, you can uncomment the following
# command for changing the source list.
#
# COPY ./docker/sources.list /etc/apt/sources.list
RUN mkdir /kaner && \
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
		python3-pip \
		python3-setuptools \
		vim \
		software-properties-common \
		dialog \
		apt-utils \
		systemd \
		language-pack-zh-hans \
		language-pack-en \
		global \
		silversearcher-ag \
		rsync \
		tmux \
		htop \
		net-tools \
		openssh-server \
		w3m \
		atop \
		iftop \
		nmon \
		nmap \
		imagemagick \
		jq \
		libpng-dev \
		zlib1g-dev \
		libpoppler-glib-dev \
		libpoppler-private-dev && \
        python3 -m pip install --upgrade pip
WORKDIR /kaner
COPY ./ /kaner/
RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["python3", "app.py", "serve"]
		