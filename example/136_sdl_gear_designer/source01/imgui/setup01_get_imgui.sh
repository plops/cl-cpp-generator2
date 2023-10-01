#!/bin/bash

#Create a bash script that downloads the following files relative to this github url:
BASE=https://raw.githubusercontent.com/ocornut/imgui/master/

# List of files to download
FILES=("imgui.h" "imgui.cpp" "imgui_draw.cpp" "imgui_internal.h" "imgui_tabes.cpp" 
       "imgui_widgets.cpp" "imstb_rectpack.h" "imstb_truetype.h" "imstb_textedit.h" 
       "backends/imgui_impl_sdl2.h" "backends/imgui_impl_sdl2.cpp" 
       "backends/imgui_impl_opengl3.h" "backends/imgui_impl_opengl3_loader.h" "backends/imgui_impl_opengl3.cpp")

# Download each file
for file in "${FILES[@]}"; do
    # Create the directory if it doesn't exist
    mkdir -p $(dirname $file)
    
    # Download the file
    curl -o $file "$BASE$file"
done

# The script will download the files into the current directory, creating any necessary subdirectories like backends/.



