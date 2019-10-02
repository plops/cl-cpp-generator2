CC=gcc
LDFLAGS=-lvulkan `pkg-config --static --libs glfw3`
CFLAGS=-march=native -ffast-math -std=gnu18 -O0 -ggdb `pkg-config --cflags cglm` -fsanitize=address #-ggdb
#CFLAGS=-std=c18 -O2 -march=native
#CFLAGS=-std=c18 -O0 -Wno-variadic-macros -W -Wpedantic -Wextra -Wall -Wcast-align -Wcast-qual -Wstrict-aliasing=2 -Wframe-larger-than=32768 -Wno-strict-overflow -Wsync-nand -Wtrampolines -Wsign-compare -Werror=float-equal -Werror=missing-braces -Werror=init-self -Werror=logical-op -Werror=write-strings -Werror=address -Werror=array-bounds -Werror=char-subscripts -Werror=enum-compare -Werror=implicit-int -Werror=empty-body -Werror=main -Werror=aggressive-loop-optimizations -Werror=nonnull -Werror=parentheses -Werror=pointer-sign -Werror=return-type -Werror=sequence-point -Werror=uninitialized -Werror=volatile-register-var -Werror=ignored-qualifiers -Werror=missing-parameter-type -Werror=old-style-declaration -Wno-error=maybe-uninitialized -Wno-unused-function -Wodr -Wformat-signedness -Wsuggest-final-types -Wsuggest-final-methods -Wno-ignored-attributes -Wno-missing-field-initializers -Wshift-overflow=2 -Wduplicated-cond -Wduplicated-branches -Werror=restrict -Wdouble-promotion -Wformat=2
CFILES:=$(shell ls | grep .c$)
OBJ:=$(CFILES:%.c=%.o)
HFILES:=$(CFILES:%.c=%.h)
PROTOS:=$(patsubst %, proto_%, $(HFILES))
%.o: %.c #proto2.h
	$(CC) -c -o $@ $< $(CFLAGS)

all: vulkan_demo
vulkan_demo: $(OBJ) 
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

vulkan_demo_small: vulkan_00_main.c vulkan_01_instance.c vulkan_02_init.c vulkan_03_glfw_window.c vulkan_04_surface.c vulkan_05_physical_device.c vulkan_06_logical_device.c vulkan_07_swap_chain.c vulkan_08_image_view.c vulkan_09_render_pass.c vulkan_10_descriptor_set_layout.c vulkan_11_graphics_pipeline.c vulkan_12_command_pool.c vulkan_13_color_resource.c vulkan_14_depth_resources.c vulkan_15_framebuffer.c  vulkan_17_texture_image_view.c vulkan_18_texture_sampler.c  vulkan_20_vertex_buffer.c vulkan_21_index_buffer.c vulkan_22_uniform_buffers.c vulkan_23_descriptor_pool.c vulkan_24_descriptor_sets.c vulkan_25_command_buffers.c vulkan_26_sync_objects.c vulkan_27_draw_frame.c vulkan_28_cleanup.c vulkan_16_texture_image.c vulkan_19_load_object.c 
	$(CC) -march=native -ffast-math -std=gnu18 -Os `pkg-config --cflags cglm` -lvulkan -lm `pkg-config --static --libs glfw3` -o $@ $^

genclean:
	rm *.c *.o vulkan*.h proto_*.h globals.h utils.h proto2.h *.spv *.frag *.vert vulkan_demo *.s

clean:
	rm *.o vulkan*.h proto_*.h proto2.h vulkan_demo *.s

proto2.h: $(PROTOS)
	echo '#ifndef PROTO2_H' > $@
	echo '#define PROTO2_H' >> $@
	cat $^ >> $@
	echo '#endif' >> $@

proto_%.h: %.c
	sh ./gen_proto2.sh $<
