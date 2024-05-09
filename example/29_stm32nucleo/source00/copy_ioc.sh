cp ~/STM32CubeIDE/workspace_1.4.0/nucleo_l476rg_dual_adc_dac/nucleo_l476rg_dual_adc_dac.ioc nucleo_l476rg_dual_adc_dac.ioc
for i in main.c stm32l4xx_it.c stm32l4xx_hal_msp.c; do cp /home/martin/STM32CubeIDE/workspace_1.4.0/nucleo_l476rg_dual_adc_dac/Core/Src/$i boilerplate;done
cd boilerplate
for i in *.c; do dos2unix $i; indent -br $i;done
