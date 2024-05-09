/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file    stm32l4xx_it.c
  * @brief   Interrupt Service Routines.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2020 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "stm32l4xx_it.h"
/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "global_log.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/* External variables --------------------------------------------------------*/
extern DMA_HandleTypeDef hdma_adc1;
extern DMA_HandleTypeDef hdma_dac_ch1;
extern DMA_HandleTypeDef hdma_usart2_tx;
extern UART_HandleTypeDef huart2;
/* USER CODE BEGIN EV */

/* USER CODE END EV */

/******************************************************************************/
/*           Cortex-M4 Processor Interruption and Exception Handlers          */
/******************************************************************************/
/**
  * @brief This function handles Non maskable interrupt.
  */
void
NMI_Handler (void)
{
  /* USER CODE BEGIN NonMaskableInt_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 27;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END NonMaskableInt_IRQn 0 */
  /* USER CODE BEGIN NonMaskableInt_IRQn 1 */

  /* USER CODE END NonMaskableInt_IRQn 1 */
}

/**
  * @brief This function handles Hard fault interrupt.
  */
void
HardFault_Handler (void)
{
  /* USER CODE BEGIN HardFault_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 26;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END HardFault_IRQn 0 */
  while (1) {
    /* USER CODE BEGIN W1_HardFault_IRQn 0 */
    /* USER CODE END W1_HardFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Memory management fault.
  */
void
MemManage_Handler (void)
{
  /* USER CODE BEGIN MemoryManagement_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 25;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END MemoryManagement_IRQn 0 */
  while (1) {
    /* USER CODE BEGIN W1_MemoryManagement_IRQn 0 */
    /* USER CODE END W1_MemoryManagement_IRQn 0 */
  }
}

/**
  * @brief This function handles Prefetch fault, memory access fault.
  */
void
BusFault_Handler (void)
{
  /* USER CODE BEGIN BusFault_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 24;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END BusFault_IRQn 0 */
  while (1) {
    /* USER CODE BEGIN W1_BusFault_IRQn 0 */
    /* USER CODE END W1_BusFault_IRQn 0 */
  }
}

/**
  * @brief This function handles Undefined instruction or illegal state.
  */
void
UsageFault_Handler (void)
{
  /* USER CODE BEGIN UsageFault_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 23;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END UsageFault_IRQn 0 */
  while (1) {
    /* USER CODE BEGIN W1_UsageFault_IRQn 0 */
    /* USER CODE END W1_UsageFault_IRQn 0 */
  }
}

/**
  * @brief This function handles System service call via SWI instruction.
  */
void
SVC_Handler (void)
{
  /* USER CODE BEGIN SVCall_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 22;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END SVCall_IRQn 0 */
  /* USER CODE BEGIN SVCall_IRQn 1 */

  /* USER CODE END SVCall_IRQn 1 */
}

/**
  * @brief This function handles Debug monitor.
  */
void
DebugMon_Handler (void)
{
  /* USER CODE BEGIN DebugMonitor_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 21;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END DebugMonitor_IRQn 0 */
  /* USER CODE BEGIN DebugMonitor_IRQn 1 */

  /* USER CODE END DebugMonitor_IRQn 1 */
}

/**
  * @brief This function handles Pendable request for system service.
  */
void
PendSV_Handler (void)
{
  /* USER CODE BEGIN PendSV_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 20;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END PendSV_IRQn 0 */
  /* USER CODE BEGIN PendSV_IRQn 1 */

  /* USER CODE END PendSV_IRQn 1 */
}

/**
  * @brief This function handles System tick timer.
  */
void
SysTick_Handler (void)
{
  /* USER CODE BEGIN SysTick_IRQn 0 */
  {
    static int count = 0;
    (count)++;
    if ((0) == (count % 1000)) {
      {
	extern TIM_HandleTypeDef htim5;
	{
	  __auto_type prim = __get_PRIMASK ();
	  __disable_irq ();
	  glog_ts[glog_count] = htim5.Instance->CNT;
	  glog_msg[glog_count] = 19;
	  (glog_count)++;
	  if ((2048) <= (glog_count)) {
	    glog_count = 0;
	  }
	  if (!(prim)) {
	    __enable_irq ();
	  }
	}
      }
    }
  }
/* USER CODE END SysTick_IRQn 0 */
  HAL_IncTick ();
  /* USER CODE BEGIN SysTick_IRQn 1 */

  /* USER CODE END SysTick_IRQn 1 */
}

/******************************************************************************/
/* STM32L4xx Peripheral Interrupt Handlers                                    */
/* Add here the Interrupt Handlers for the used peripherals.                  */
/* For the available peripheral interrupt handler names,                      */
/* please refer to the startup file (startup_stm32l4xx.s).                    */
/******************************************************************************/

/**
  * @brief This function handles DMA1 channel1 global interrupt.
  */
void
DMA1_Channel1_IRQHandler (void)
{
  /* USER CODE BEGIN DMA1_Channel1_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 16;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END DMA1_Channel1_IRQn 0 */
  HAL_DMA_IRQHandler (&hdma_adc1);
  /* USER CODE BEGIN DMA1_Channel1_IRQn 1 */

  /* USER CODE END DMA1_Channel1_IRQn 1 */
}

/**
  * @brief This function handles DMA1 channel3 global interrupt.
  */
void
DMA1_Channel3_IRQHandler (void)
{
  /* USER CODE BEGIN DMA1_Channel3_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 17;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END DMA1_Channel3_IRQn 0 */
  HAL_DMA_IRQHandler (&hdma_dac_ch1);
  /* USER CODE BEGIN DMA1_Channel3_IRQn 1 */

  /* USER CODE END DMA1_Channel3_IRQn 1 */
}

/**
  * @brief This function handles DMA1 channel7 global interrupt.
  */
void
DMA1_Channel7_IRQHandler (void)
{
  /* USER CODE BEGIN DMA1_Channel7_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 14;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END DMA1_Channel7_IRQn 0 */
  HAL_DMA_IRQHandler (&hdma_usart2_tx);
  /* USER CODE BEGIN DMA1_Channel7_IRQn 1 */

  /* USER CODE END DMA1_Channel7_IRQn 1 */
}

/**
  * @brief This function handles USART2 global interrupt.
  */
void
USART2_IRQHandler (void)
{
  /* USER CODE BEGIN USART2_IRQn 0 */
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 13;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END USART2_IRQn 0 */
  HAL_UART_IRQHandler (&huart2);
  /* USER CODE BEGIN USART2_IRQn 1 */

  /* USER CODE END USART2_IRQn 1 */
}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */
/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
