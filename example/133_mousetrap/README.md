# Install

```

sudo emerge -av gui-libs/libadwaita media-libs/glm media-libs/glew
# 
# glm 3.2MB installed
# glew 0.7MB

cd ~/src
git clone https://github.com/clemapfel/mousetrap
cd mousetrap
mkdir b; cd b;
cmake .. \
  -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DMOUSETRAP_ENABLE_OPENGL_COMPONENT=True \
  -DCMAKE_INSTALL_PREFIX=/home/martin/moustrap

```

# References

https://news.ycombinator.com/item?id=37194317