/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
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

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <math.h>
#include <pb_encode.h>
#include <pb_decode.h>
#include "simple.pb.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;
DMA_HandleTypeDef hdma_adc1;

DAC_HandleTypeDef hdac1;
DMA_HandleTypeDef hdma_dac_ch1;

TIM_HandleTypeDef htim2;
TIM_HandleTypeDef htim4;
TIM_HandleTypeDef htim5;
TIM_HandleTypeDef htim6;

UART_HandleTypeDef huart2;
DMA_HandleTypeDef hdma_usart2_tx;

/* USER CODE BEGIN PV */
uint32_t glog_ts[2048];
uint8_t glog_msg[2048];
int glog_count;
uint16_t value_adc[60];
uint16_t value_dac[60];
uint8_t BufferToSend[512];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config (void);
static void MX_GPIO_Init (void);
static void MX_DMA_Init (void);
static void MX_USART2_UART_Init (void);
static void MX_ADC1_Init (void);
static void MX_TIM2_Init (void);
static void MX_TIM5_Init (void);
static void MX_DAC1_Init (void);
static void MX_TIM6_Init (void);
static void MX_TIM4_Init (void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
bool
encode_int32 (pb_ostream_t * stream, const pb_field_t * field,
	      void *const *arg)
{
  for (__auto_type i = 0; (i) < (10); (i) += (1)) {
    if (!(pb_encode_tag_for_field (stream, field))) {
      return false;
    }
    if (!(pb_encode_varint (stream, ((i) + (42))))) {
      return false;
    }
  }
  return true;
}

void
HAL_ADC_ConvHalfCpltCallback (ADC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 0;
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

void
HAL_ADC_ErrorCallback (ADC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 1;
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

void
HAL_ADC_ConvCpltCallback (ADC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 2;
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

void
HAL_UART_ErrorCallback (UART_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 3;
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

void
HAL_UART_TransmitCpltCallback (UART_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 4;
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

void
HAL_UART_AbortOnErrorCallback (UART_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 5;
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

void
HAL_DAC_ErrorCallbackCh1 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 6;
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

void
HAL_DAC_ConvCpltCallbackCh1 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 7;
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

void
HAL_DAC_ConvHalfCpltCallbackCh1 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 8;
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

void
HAL_DAC_ErrorCallbackCh2 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 9;
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

void
HAL_DAC_ConvCpltCallbackCh2 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 10;
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

void
HAL_DAC_ConvHalfCpltCallbackCh2 (DAC_HandleTypeDef * arg)
{
  const int output_p = 1;
  // no counter
  ;
  if (output_p) {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 11;
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

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int
main (void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration-------------------------------------------------------- */

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init ();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config ();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init ();
  MX_DMA_Init ();
  MX_USART2_UART_Init ();
  MX_ADC1_Init ();
  MX_TIM2_Init ();
  MX_TIM5_Init ();
  MX_DAC1_Init ();
  MX_TIM6_Init ();
  MX_TIM4_Init ();
  /* USER CODE BEGIN 2 */
  HAL_TIM_Base_Init (&htim6);
  HAL_TIM_Base_Start (&htim6);
  HAL_TIM_Base_Init (&htim4);
  HAL_TIM_Base_Start (&htim4);
  HAL_TIM_PWM_Start (&htim4, TIM_CHANNEL_1);
  HAL_TIM_Base_Init (&htim2);
  HAL_TIM_Base_Start (&htim2);
  HAL_TIM_PWM_Start (&htim2, TIM_CHANNEL_1);
  HAL_TIM_PWM_Start (&htim2, TIM_CHANNEL_2);
  HAL_TIM_Base_Init (&htim5);
  HAL_TIM_Base_Start (&htim5);
  for (__auto_type i = 0; (i) < (60); (i) += (1)) {
    __auto_type v = 0;
    value_dac[i] = v;
  }
  value_dac[0] = 4095;
  value_dac[1] = 0;
  value_dac[2] = 0;
  value_dac[3] = 4095;
  value_dac[4] = 0;
  value_dac[5] = 0;
  value_dac[6] = 0;
  value_dac[7] = 4095;
  HAL_DAC_Init (&hdac1);
  HAL_DAC_Start (&hdac1, DAC_CHANNEL_1);
  HAL_DAC_Start_DMA (&hdac1, DAC_CHANNEL_1, (uint32_t *) value_dac, 60,
		     DAC_ALIGN_12B_R);
  HAL_ADCEx_Calibration_Start (&hadc1, ADC_SINGLE_ENDED);
  HAL_ADC_Start_DMA (&hadc1, (uint32_t *) value_adc, 60);
  {
    extern TIM_HandleTypeDef htim5;
    {
      __auto_type prim = __get_PRIMASK ();
      __disable_irq ();
      glog_ts[glog_count] = htim5.Instance->CNT;
      glog_msg[glog_count] = 12;
      (glog_count)++;
      if ((2048) <= (glog_count)) {
	glog_count = 0;
      }
      if (!(prim)) {
	__enable_irq ();
      }
    }
  }
/* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    {
      static int count;
      (count)++;
      if ((4096) <= (count)) {
	count = 0;
      }
      HAL_Delay (30);
      (htim2.Instance->CCR2)++;
      if ((77) <= (htim2.Instance->CCR2)) {
	htim2.Instance->CCR2 = 2;
      }
      {
	BufferToSend[0] = 85;
	BufferToSend[1] = 85;
	BufferToSend[2] = 85;
	BufferToSend[3] = 85;
	BufferToSend[4] = 85;
	SimpleMessage message = SimpleMessage_init_zero;
	__auto_type stream =
	  pb_ostream_from_buffer (((5) + (2) + (BufferToSend)),
				  ((sizeof (BufferToSend)) - (5)));
	message.id = 1431655765;
	message.timestamp = htim5.Instance->CNT;
	message.phase = htim2.Instance->CCR2;
	message.int32value.funcs.encode = &encode_int32;
	message.sample00 = value_adc[0];
	message.sample01 = value_adc[1];
	message.sample02 = value_adc[2];
	message.sample03 = value_adc[3];
	message.sample04 = value_adc[4];
	message.sample05 = value_adc[5];
	message.sample06 = value_adc[6];
	message.sample07 = value_adc[7];
	message.sample08 = value_adc[8];
	message.sample09 = value_adc[9];
	message.sample10 = value_adc[10];
	message.sample11 = value_adc[11];
	message.sample12 = value_adc[12];
	message.sample13 = value_adc[13];
	message.sample14 = value_adc[14];
	message.sample15 = value_adc[15];
	message.sample16 = value_adc[16];
	message.sample17 = value_adc[17];
	message.sample18 = value_adc[18];
	message.sample19 = value_adc[19];
	message.sample20 = value_adc[20];
	message.sample21 = value_adc[21];
	message.sample22 = value_adc[22];
	message.sample23 = value_adc[23];
	message.sample24 = value_adc[24];
	message.sample25 = value_adc[25];
	message.sample26 = value_adc[26];
	message.sample27 = value_adc[27];
	message.sample28 = value_adc[28];
	message.sample29 = value_adc[29];
	message.sample30 = value_adc[30];
	message.sample31 = value_adc[31];
	message.sample32 = value_adc[32];
	message.sample33 = value_adc[33];
	message.sample34 = value_adc[34];
	message.sample35 = value_adc[35];
	message.sample36 = value_adc[36];
	message.sample37 = value_adc[37];
	message.sample38 = value_adc[38];
	message.sample39 = value_adc[39];
	message.sample40 = value_adc[40];
	message.sample41 = value_adc[41];
	message.sample42 = value_adc[42];
	message.sample43 = value_adc[43];
	message.sample44 = value_adc[44];
	message.sample45 = value_adc[45];
	message.sample46 = value_adc[46];
	message.sample47 = value_adc[47];
	message.sample48 = value_adc[48];
	message.sample49 = value_adc[49];
	message.sample50 = value_adc[50];
	message.sample51 = value_adc[51];
	message.sample52 = value_adc[52];
	message.sample53 = value_adc[53];
	message.sample54 = value_adc[54];
	message.sample55 = value_adc[55];
	message.sample56 = value_adc[56];
	message.sample57 = value_adc[57];
	message.sample58 = value_adc[58];
	message.sample59 = value_adc[59];
	__auto_type status =
	  pb_encode (&stream, SimpleMessage_fields, &message);
	__auto_type message_length = stream.bytes_written;
	if (status) {
	  BufferToSend[((5) + (0))] = ((255) & (message_length));
	  BufferToSend[((5) + (1))] = (((65280) & (message_length))) >> (8);
	  BufferToSend[((5) + (2) + (message_length) + (0))] = 255;
	  BufferToSend[((5) + (2) + (message_length) + (1))] = 255;
	  BufferToSend[((5) + (2) + (message_length) + (2))] = 255;
	  BufferToSend[((5) + (2) + (message_length) + (3))] = 255;
	  BufferToSend[((5) + (2) + (message_length) + (4))] = 255;
	  if (!
	      ((HAL_OK) ==
	       (HAL_UART_Transmit_DMA
		(&huart2, (uint8_t *) BufferToSend,
		 ((5) + (2) + (message_length) + (5)))))) {
	    Error_Handler ();
	  }
	}
      }
    }
  }
/* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void
SystemClock_Config (void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = { 0 };
  RCC_ClkInitTypeDef RCC_ClkInitStruct = { 0 };
  RCC_PeriphCLKInitTypeDef PeriphClkInit = { 0 };

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 10;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV7;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig (&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler ();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK
    | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig (&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
    Error_Handler ();
  }
  PeriphClkInit.PeriphClockSelection =
    RCC_PERIPHCLK_USART2 | RCC_PERIPHCLK_ADC;
  PeriphClkInit.Usart2ClockSelection = RCC_USART2CLKSOURCE_PCLK1;
  PeriphClkInit.AdcClockSelection = RCC_ADCCLKSOURCE_SYSCLK;
  if (HAL_RCCEx_PeriphCLKConfig (&PeriphClkInit) != HAL_OK) {
    Error_Handler ();
  }
  HAL_RCC_MCOConfig (RCC_MCO1, RCC_MCO1SOURCE_SYSCLK, RCC_MCODIV_16);
  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling (PWR_REGULATOR_VOLTAGE_SCALE1) !=
      HAL_OK) {
    Error_Handler ();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_ADC1_Init (void)
{

  /* USER CODE BEGIN ADC1_Init 0 */
  {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 48;
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
/* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = { 0 };
  ADC_ChannelConfTypeDef sConfig = { 0 };

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */
  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_12B;
  hadc1.Init.DataAlign = ADC_DATAALIGN_RIGHT;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_EXTERNALTRIG_T2_CC2;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_RISING;
  hadc1.Init.DMAContinuousRequests = ENABLE;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init (&hadc1) != HAL_OK) {
    Error_Handler ();
  }
  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel (&hadc1, &multimode) != HAL_OK) {
    Error_Handler ();
  }
  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_1;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_6CYCLES_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  if (HAL_ADC_ConfigChannel (&hadc1, &sConfig) != HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief DAC1 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_DAC1_Init (void)
{

  /* USER CODE BEGIN DAC1_Init 0 */
  {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 47;
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
/* USER CODE END DAC1_Init 0 */

  DAC_ChannelConfTypeDef sConfig = { 0 };

  /* USER CODE BEGIN DAC1_Init 1 */

  /* USER CODE END DAC1_Init 1 */
  /** DAC Initialization
  */
  hdac1.Instance = DAC1;
  if (HAL_DAC_Init (&hdac1) != HAL_OK) {
    Error_Handler ();
  }
  /** DAC channel OUT1 config
  */
  sConfig.DAC_SampleAndHold = DAC_SAMPLEANDHOLD_DISABLE;
  sConfig.DAC_Trigger = DAC_TRIGGER_T4_TRGO;
  sConfig.DAC_OutputBuffer = DAC_OUTPUTBUFFER_DISABLE;
  sConfig.DAC_ConnectOnChipPeripheral = DAC_CHIPCONNECT_DISABLE;
  sConfig.DAC_UserTrimming = DAC_TRIMMING_FACTORY;
  if (HAL_DAC_ConfigChannel (&hdac1, &sConfig, DAC_CHANNEL_1) != HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN DAC1_Init 2 */

  /* USER CODE END DAC1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_TIM2_Init (void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = { 0 };
  TIM_MasterConfigTypeDef sMasterConfig = { 0 };
  TIM_OC_InitTypeDef sConfigOC = { 0 };

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 0;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 79;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init (&htim2) != HAL_OK) {
    Error_Handler ();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource (&htim2, &sClockSourceConfig) != HAL_OK) {
    Error_Handler ();
  }
  if (HAL_TIM_PWM_Init (&htim2) != HAL_OK) {
    Error_Handler ();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization (&htim2, &sMasterConfig) !=
      HAL_OK) {
    Error_Handler ();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 40;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel (&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
    Error_Handler ();
  }
  if (HAL_TIM_PWM_ConfigChannel (&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit (&htim2);

}

/**
  * @brief TIM4 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_TIM4_Init (void)
{

  /* USER CODE BEGIN TIM4_Init 0 */

  /* USER CODE END TIM4_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = { 0 };
  TIM_MasterConfigTypeDef sMasterConfig = { 0 };
  TIM_OC_InitTypeDef sConfigOC = { 0 };

  /* USER CODE BEGIN TIM4_Init 1 */

  /* USER CODE END TIM4_Init 1 */
  htim4.Instance = TIM4;
  htim4.Init.Prescaler = 0;
  htim4.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim4.Init.Period = 79;
  htim4.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim4.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init (&htim4) != HAL_OK) {
    Error_Handler ();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource (&htim4, &sClockSourceConfig) != HAL_OK) {
    Error_Handler ();
  }
  if (HAL_TIM_PWM_Init (&htim4) != HAL_OK) {
    Error_Handler ();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization (&htim4, &sMasterConfig) !=
      HAL_OK) {
    Error_Handler ();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 40;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel (&htim4, &sConfigOC, TIM_CHANNEL_1) != HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN TIM4_Init 2 */

  /* USER CODE END TIM4_Init 2 */
  HAL_TIM_MspPostInit (&htim4);

}

/**
  * @brief TIM5 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_TIM5_Init (void)
{

  /* USER CODE BEGIN TIM5_Init 0 */

  /* USER CODE END TIM5_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = { 0 };
  TIM_MasterConfigTypeDef sMasterConfig = { 0 };

  /* USER CODE BEGIN TIM5_Init 1 */

  /* USER CODE END TIM5_Init 1 */
  htim5.Instance = TIM5;
  htim5.Init.Prescaler = 0;
  htim5.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim5.Init.Period = 4294967295;
  htim5.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim5.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init (&htim5) != HAL_OK) {
    Error_Handler ();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource (&htim5, &sClockSourceConfig) != HAL_OK) {
    Error_Handler ();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization (&htim5, &sMasterConfig) !=
      HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN TIM5_Init 2 */

  /* USER CODE END TIM5_Init 2 */

}

/**
  * @brief TIM6 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_TIM6_Init (void)
{

  /* USER CODE BEGIN TIM6_Init 0 */

  /* USER CODE END TIM6_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = { 0 };

  /* USER CODE BEGIN TIM6_Init 1 */

  /* USER CODE END TIM6_Init 1 */
  htim6.Instance = TIM6;
  htim6.Init.Prescaler = 0;
  htim6.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim6.Init.Period = 65535;
  htim6.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init (&htim6) != HAL_OK) {
    Error_Handler ();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_UPDATE;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization (&htim6, &sMasterConfig) !=
      HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN TIM6_Init 2 */

  /* USER CODE END TIM6_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void
MX_USART2_UART_Init (void)
{

  /* USER CODE BEGIN USART2_Init 0 */
  {
    {
      extern TIM_HandleTypeDef htim5;
      {
	__auto_type prim = __get_PRIMASK ();
	__disable_irq ();
	glog_ts[glog_count] = htim5.Instance->CNT;
	glog_msg[glog_count] = 46;
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
/* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_8;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_ENABLE;
  huart2.AdvancedInit.AdvFeatureInit =
    UART_ADVFEATURE_RXOVERRUNDISABLE_INIT |
    UART_ADVFEATURE_DMADISABLEONERROR_INIT;
  huart2.AdvancedInit.OverrunDisable = UART_ADVFEATURE_OVERRUN_DISABLE;
  huart2.AdvancedInit.DMADisableonRxError =
    UART_ADVFEATURE_DMA_DISABLEONRXERROR;
  if (HAL_HalfDuplex_Init (&huart2) != HAL_OK) {
    Error_Handler ();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void
MX_DMA_Init (void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE ();

  /* DMA interrupt init */
  /* DMA1_Channel1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority (DMA1_Channel1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ (DMA1_Channel1_IRQn);
  /* DMA1_Channel3_IRQn interrupt configuration */
  HAL_NVIC_SetPriority (DMA1_Channel3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ (DMA1_Channel3_IRQn);
  /* DMA1_Channel7_IRQn interrupt configuration */
  HAL_NVIC_SetPriority (DMA1_Channel7_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ (DMA1_Channel7_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void
MX_GPIO_Init (void)
{
  GPIO_InitTypeDef GPIO_InitStruct = { 0 };

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE ();
  __HAL_RCC_GPIOH_CLK_ENABLE ();
  __HAL_RCC_GPIOA_CLK_ENABLE ();
  __HAL_RCC_GPIOB_CLK_ENABLE ();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin (LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init (B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init (LD2_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PA8 */
  GPIO_InitStruct.Pin = GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  GPIO_InitStruct.Alternate = GPIO_AF0_MCO;
  HAL_GPIO_Init (GPIOA, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void
Error_Handler (void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */

  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void
assert_failed (uint8_t * file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     tex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
