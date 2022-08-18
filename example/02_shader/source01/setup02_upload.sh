cat main.cpp | sed 's/^[ \t]*//' > main_trim.cpp
scp main_trim.cpp hetzner:/dev/shm/
# upload with python script from cl-py-generator/example/95_shadertoy
ssh hetzner /home/martin/st.sh /dev/shm/main_trim.cpp
