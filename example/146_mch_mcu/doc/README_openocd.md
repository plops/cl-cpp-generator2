Here is the English translation (using Google Gemini Advanced 1.0) of the MRS Linux _x64 Toolchain & OpenOCD User Manual:

## 1. FLASH CMD

**[RISC-V]**

* Erase all sectors:
```
sudo ./openocd -f wch-riscv.cfg -c init -c halt -c "flash erase_sector wch_riscv 0 last " -c exit
```

* Erase all sectors and program:
```
sudo ./openocd -f wch-riscv.cfg -c init -c halt -c "program xxx.hex\bin\elf " -c exit
```

* Verify image:
```
sudo ./openocd -f wch-riscv.cfg -c init -c halt -c "verify_image xxx.hex\bin\elf"  -c exit
```

* Reset and resume:
```
sudo ./openocd -f wch-riscv.cfg -c init -c halt -c wlink_reset_resume  -c exit
```

**[ARM]**

* Erase all sectors:
```
sudo ./openocd -f wch-arm.cfg -c init -c halt -c "flash erase_sector wch_arm 0 last " -c exit
```

* Erase all sectors and program:
```
sudo ./openocd -f wch-arm.cfg -c init -c halt -c "program xxx.hex\bin\elf " -c exit
```

* Verify image:
```
sudo ./openocd -f wch-arm.cfg -c init -c halt -c "verify_image xxx.hex\bin\elf"  -c exit
```

## 2. GDB CMD

1. Create an OpenOCD process:
```
sudo ./[OPENOCD_PATH]/openocd -f [CFGFILE_PATH]
```

2. Start a GDB process:
```
sudo ./[GDB_PATH]/riscv-none-embed-gdb
```

3. GDB commands (must be executed in sequence):

* Specify the debug file:
```
file [FILE_PATH]
```

* Connect to the port:
```
target remote localhost:3333
```

* Program the code:
```
load
```

* View registers:
```
info register [REGISTER_NAME]
```

* View the current pc value:
```
i r pc
```

* View breakpoint information:
```
info breakpoint
```

* Set a breakpoint:
```
break [LINE_NUM/FUNC_NAME/SYMBOL_NAME]
```

* Continue running:
```
continue
```

* Step next:
```
next
```

* Step into:
```
step
```

* Print variable value:
```
print
```

* View current code:
```
list (requires the project directory to contain the source code, and the debug level is -g or higher when compiling)
```

* In the paused state, you can perform operations such as viewing registers, viewing current code, and viewing disassembly.

## 3. Debugging Step Demonstration

1. Create an openocd process

Format: OPENOCD_PATH -f CFG_PATH
If you are on the Ubuntu platform, it is recommended to use the drag and drop method to improve efficiency and avoid input path errors.

After pressing Enter, the openocd process hangs and waits for connection. The port number waiting for connection is displayed, which is 3333 in this case.

cd openocd path

Command:

sudo ./openocd -f wch-riscv.cfg


2. Start the gdb process. Specify the debug elf

Format: GDB_PATH [FILE_PATH] [–ARGS]

Without parameters, riscv-none-embed-gdb. By default, gdb cli commands are supported. If FILE_PATH is not specified, the file command needs to be used to specify the debug file later.

riscv-none-embed-gdb xxxx.elf with debug file, no need to specify file later

riscv-none-embed-gdb xxxx.elf–interpreter mi supports gdb mi commands in addition to cli. After MRS debugging turns on gdb trace, the commands output in Console are gdb mi commands. You can copy them to this mode and run them one by one.

If there are no parameters, and you need to specify the debug file later, the command is file FILE_PATH

Open another terminal,

Command

cd  riscv-none-embed-gdb path

Sudo ./riscv-none-embed-gdb 

file +xxx.elf (elf file directory + file
