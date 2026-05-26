#!/usr/bin/env bash

# command line arguments, expect environment name and user name to setup in this script
PYTHON_ENV=$1
USERNAME=$2

# create python virtual environment and set path to use
python3 -m venv /opt/$PYTHON_ENV  
export PATH=/opt/$PYTHON_ENV/bin:$PATH

# update root and user bashrc files to correctly setup environment on reboots
# this is run as root, so first update root bashrc environment settings
cat >> /root/.bashrc << EOF

# the virtual environment used for this assignment
PYTHON_ENV=$PYTHON_ENV
EOF

cat >> /root/.bashrc <<'EOF'

# search for nvidia libraries and add to LD_LIBRARY_PATH if
# nvidia / cuda support is installed
LIBS=$(find /opt/$PYTHON_ENV -name "*.so*" | grep nvidia)
if [ -n "$LIBS" ];
then
  export LD_LIBRARY_PATH=$(echo $LIBS | xargs dirname | sort -u | paste -d ":" -s -)
fi

# activate the default virtual environment for assignments
source /opt/$PYTHON_ENV/bin/activate

EOF

# do for the normal user, this is duplicate setup, is there a way to specify a heredoc without repeating like this?
cat >> /home/$USERNAME/.bashrc << EOF

# the virtual environment used for this assignment
PYTHON_ENV=$PYTHON_ENV
EOF

cat >> /home/$USERNAME/.bashrc <<'EOF'

# search for nvidia libraries and add to LD_LIBRARY_PATH if
# nvidia / cuda support is installed
LIBS=$(find /opt/$PYTHON_ENV -name "*.so*" | grep nvidia)
if [ -n "$LIBS" ];
then
  export LD_LIBRARY_PATH=$(echo $LIBS | xargs dirname | sort -u | paste -d ":" -s -)
fi

# activate the default virtual environment for assignments
source /opt/$PYTHON_ENV/bin/activate

EOF


# install all required packages into virtual environment specified in requirements.txt
source /opt/$PYTHON_ENV/bin/activate
pip3 install -q -r ./requirements/requirements.txt