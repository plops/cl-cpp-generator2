project('risc_test', ['c', 'cpp'], default_options: ['cpp_std=c++20'])

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
    '--specs=nosys.specs',
    '-Wl,-gc-sections,--print-gc-sections,--print-memory-usage',
    '-T../../hal/Ld/Link.ld',
    '../../hal/StdPeriphDriver/libISP592.a',
    '../../BLE/LIB/LIBCH59xBLE.a'
]


add_project_arguments(riscv_flags, language: 'c')
add_project_arguments(riscv_flags, language: 'cpp')

mine_src = ['src/main.cpp']


# HAL sources
# use glob to include all files in a directory

hal_sources = files(
    '../hal/StdPeriphDriver/CH59x_sys.c',
    '../hal/StdPeriphDriver/CH59x_clk.c',
    '../hal/StdPeriphDriver/CH59x_adc.c',
    '../hal/StdPeriphDriver/CH59x_pwr.c',
    '../hal/StdPeriphDriver/CH59x_gpio.c',
    '../hal/RVMSIS/core_riscv.c',
    '../hal/Startup/startup_CH592.S',
    '../hal/User/board.c',
    '../BLE/HAL/KEY.c',
    '../BLE/HAL/LED.c',
    '../BLE/HAL/MCU.c',
    '../BLE/HAL/RTC.c',
    '../BLE/HAL/SLEEP.c',
    '../BLE/LIB/ble_task_scheduler.S'
)


# Debug/ Release flags
buildtype = get_option('buildtype')
if buildtype == 'debug'
  debug_flags = ['-g', '-g3', '-ggdb', '-gdwarf-4']
else
  optimization_flags = ['-O3']
  riscv_link_args += [
    '-Wl,-gc-sections,--print-gc-sections,--print-memory-usage',
    #'-Map=' + meson.build_root() + '/' + project_name + '.map'
  ]
endif

# Executable linking it all together
executable('risc_test',
           sources : hal_sources + mine_src,
           include_directories : include_directories(
               '../hal/RVMSIS',
               '../hal/StdPeriphDriver/inc',
               '../hal/User',
               '../BLE/HAL/include',
               '../BLE/LIB'
           ),
           #link_with : [
           #    '../hal/StdPeriphDriver/libISP592.a',
              # '../BLE/LIB/LIBCH59xBLE.a',
           #],
           cpp_args : riscv_flags + optimization_flags,
           link_args : riscv_link_args   # Custom linker script
)
