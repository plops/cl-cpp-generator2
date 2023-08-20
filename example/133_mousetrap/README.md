# Install

```

sudo emerge -av gui-libs/libadwaita media-libs/glm
# 
# glm 3.2MB installed

cd ~/src
git clone https://github.com/clemapfel/mousetrap
cd mousetrap
mkdir b; cd b;
cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release 

```