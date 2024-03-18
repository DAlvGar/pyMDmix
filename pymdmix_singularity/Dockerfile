FROM ggutierrezbio/ambertools20
ARG CONDA_PREFIX="/opt/SOFT/miniconda3"
ENV PATH="$CONDA_PREFIX/bin:${PATH}"
ARG PATH="$CONDA_PREFIX/bin:${PATH}"

SHELL [ "/bin/bash", "--login", "-c" ]
RUN apt-get update && apt-get install -y wget git python2.7 python-dev curl && rm -rf /var/lib/apt/lists/*
RUN curl https://bootstrap.pypa.io/pip/2.7/get-pip.py --output get-pip.py
RUN python2.7 get-pip.py
RUN pip install -e git://github.com/CBDD/pyMDmix.git@master-legacy-fix#egg=pymdmix
RUN chmod -R o+rX /src
RUN cp /src/pymdmix/pyMDMix/data/defaults/settings.cfg /src/pymdmix/pyMDMix/data/settings.cfg
RUN cp /src/pymdmix/pyMDMix/data/defaults/md-settings.cfg /src/pymdmix/pyMDMix/data/md-settings.cfg
RUN cp /src/pymdmix/pyMDMix/data/solventlib/SOLVENTS.db /src/pymdmix/pyMDMix/data/SOLVENTS.db
ADD daint_queue_temp.txt /src/pymdmix/pyMDMix/data/templates/
ENTRYPOINT ["mdmix"]
