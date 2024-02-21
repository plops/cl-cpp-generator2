project('risc_test', ['c', 'cpp', 'asm'], default_options: ['cpp_std=c++20'])

# Compiler/linker options 
riscv_flags = [
    '-march=rv32imac_zicsr_zifencei',
    '-mabi=ilp32',
    '-msmall-data-limit=8',
    '-mno-save-restore'
]

riscv_link_args = [ 
    '-march=rv32imac_zicsr_zifencei',
    '-mabi=ilp32',
    '-Xlinker', '--gc-sections',
    '--specs=nano.specs',
    '--specs=nosys.specs'
]

# Collect ALL source files
all_sources = files(
    'src',                                    # Your source directory
    '../hal/StdPeriphDriver', 
    '../hal/RVMSIS',
    '../hal/Startup',
    '../hal/board.c',
    '../hal/BLE/HAL',
    '../hal/BLE/LIB'
)

# Debug/ Release flags
buildtype = get_option('buildtype')
if buildtype == 'debug'
  debug_flags = ['-g', '-g3', '-ggdb', '-gdwarf-4']
else
  optimization_flags = ['-O3']
  riscv_link_args += [
    '-Wl,-gc-sections,--print-gc-sections,--print-memory-usage',
    '-Map=' + meson.build_root() + '/' + project_name + '.map'
  ]
endif

# Executable linking it all together
executable('risc_test',
           sources : all_sources,
           include_directories : include_directories(
               '../hal/RVMSIS',
               '../hal/StdPeriphDriver/inc',
               '../hal/User',
               '../hal/BLE/HAL/include',
               '../hal/BLE/LIB'
           ),
           link_with : [
               '../hal/StdPeriphDriver/libISP592.a',
               '../hal/BLE/LIB/LIBCH59xBLE.a',
           ],
           cpp_args : riscv_flags + debug_flags + optimization_flags,
           link_args : riscv_link_args,
           link_args : '-T../hal/Ld/Link.ld'  # Custom linker script
)
