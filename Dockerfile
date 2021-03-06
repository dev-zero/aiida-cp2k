###############################################################################
# Copyright (c), The AiiDA-CP2K authors.                                      #
# SPDX-License-Identifier: MIT                                                #
# AiiDA-CP2K is hosted on GitHub at https://github.com/aiidateam/aiida-cp2k   #
# For further information on the license, see the LICENSE.txt file.           #
###############################################################################

FROM ubuntu:rolling
USER root

# silence tzdata's setup dialog
ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN=true

RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential       \
    git                   \
    python-setuptools     \
    python-wheel          \
    python-pip            \
    python-dev            \
    postgresql            \
    rabbitmq-server       \
    less                  \
    nano                  \
    sudo                  \
    ssh                   \
    cp2k                  \
    python3               \
    python3-setuptools    \
  && rm -rf /var/lib/apt/lists/*

RUN pip install codecov pytest-cov twine

# set a unicode-enabled locale by default, and make sure the locale files are available
RUN set -eux; \
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# create ubuntu user with sudo powers
RUN adduser --disabled-password --gecos "" ubuntu \
    && echo "ubuntu ALL=(ALL) NOPASSWD: ALL" >>  /etc/sudoers

# install aiida-cp2k
COPY --chown=ubuntu:ubuntu . /opt/aiida-cp2k
WORKDIR /opt/aiida-cp2k/
RUN pip install .[pre-commit,testing]

# Fix pgtest, see https://github.com/jamesnunn/pgtest/issues/14
RUN sed -i -e 's|\(max_connections\)=11|\1=15|' /usr/local/lib/python2.7/dist-packages/pgtest/pgtest.py

# switch to an unprivileged user
USER ubuntu
