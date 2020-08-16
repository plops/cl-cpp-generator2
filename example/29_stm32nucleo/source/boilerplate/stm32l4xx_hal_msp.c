/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * File Name          : stm32l4xx_hal_msp.c
  * Description        : This file provides code for the MSP Initialization
  *                      and de-Initialization codes.
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
/* USER CODE BEGIN Includes */
#include "global_log.h"
/* USER CODE END Includes */
extern DMA_HandleTypeDef hdma_adc1;

extern DMA_HandleTypeDef hdma_dac_ch1;

extern DMA_HandleTypeDef hdma_usart2_tx;

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN TD */

/* USER CODE END TD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN Define */

/* USER CODE END Define */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN Macro */

/* USER CODE END Macro */

/* Private variables ---------------------------------------------------------*/
/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* External functions --------------------------------------------------------*/
/* USER CODE BEGIN ExternalFunctions */

/* USER CODE END ExternalFunctions */

/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);
                                        /**
  * Initializes the Global MSP.
  */
void HAL_MspInit(void)
{
  /* USER CODE BEGIN MspInit 0 */

  /* USER CODE END MspInit 0 */

  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_RCC_PWR_CLK_ENABLE();

  /* System interrupt init*/

  /* USER CODE BEGIN MspInit 1 */

  /* USER CODE END MspInit 1 */
}

/**
* @brief ADC MSP Initialization
* This function configures the hardware resources used in this example
* @param hadc: ADC handle pointer
* @retval None
*/
void HAL_ADC_MspInit(ADC_HandleTypeDef* hadc)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(hadc->Instance==ADC1)
  {
  /* USER CODE BEGIN ADC1_MspInit 0 */

  /* USER CODE END ADC1_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_ADC_CLK_ENABLE();

    __HAL_RCC_GPIOC_CLK_ENABLE();
    /**ADC1 GPIO Configuration
    PC0     ------> ADC1_IN1
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG_ADC_CONTROL;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    /* ADC1 DMA Init */
    /* ADC1 Init */
    hdma_adc1.Instance = DMA1_Channel1;
    hdma_adc1.Init.Request = DMA_REQUEST_0;
    hdma_adc1.Init.Direction = DMA_PERIPH_TO_MEMORY;
    hdma_adc1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_adc1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_adc1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_adc1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
    hdma_adc1.Init.Mode = DMA_CIRCULAR;
    hdma_adc1.Init.Priority = DMA_PRIORITY_LOW;
    if (HAL_DMA_Init(&hdma_adc1) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(hadc,DMA_Handle,hdma_adc1);

  /* USER CODE BEGIN ADC1_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=32;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END ADC1_MspInit 1 */
  }

}

/**
* @brief ADC MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param hadc: ADC handle pointer
* @retval None
*/
void HAL_ADC_MspDeInit(ADC_HandleTypeDef* hadc)
{
  if(hadc->Instance==ADC1)
  {
  /* USER CODE BEGIN ADC1_MspDeInit 0 */

  /* USER CODE END ADC1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_ADC_CLK_DISABLE();

    /**ADC1 GPIO Configuration
    PC0     ------> ADC1_IN1
    */
    HAL_GPIO_DeInit(GPIOC, GPIO_PIN_0);

    /* ADC1 DMA DeInit */
    HAL_DMA_DeInit(hadc->DMA_Handle);
  /* USER CODE BEGIN ADC1_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=33;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END ADC1_MspDeInit 1 */
  }

}

/**
* @brief DAC MSP Initialization
* This function configures the hardware resources used in this example
* @param hdac: DAC handle pointer
* @retval None
*/
void HAL_DAC_MspInit(DAC_HandleTypeDef* hdac)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(hdac->Instance==DAC1)
  {
  /* USER CODE BEGIN DAC1_MspInit 0 */

  /* USER CODE END DAC1_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_DAC1_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**DAC1 GPIO Configuration
    PA4     ------> DAC1_OUT1
    */
    GPIO_InitStruct.Pin = GPIO_PIN_4;
    GPIO_InitStruct.Mode = GPIO_MODE_ANALOG;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    /* DAC1 DMA Init */
    /* DAC_CH1 Init */
    hdma_dac_ch1.Instance = DMA1_Channel3;
    hdma_dac_ch1.Init.Request = DMA_REQUEST_6;
    hdma_dac_ch1.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_dac_ch1.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_dac_ch1.Init.MemInc = DMA_MINC_ENABLE;
    hdma_dac_ch1.Init.PeriphDataAlignment = DMA_PDATAALIGN_HALFWORD;
    hdma_dac_ch1.Init.MemDataAlignment = DMA_MDATAALIGN_HALFWORD;
    hdma_dac_ch1.Init.Mode = DMA_CIRCULAR;
    hdma_dac_ch1.Init.Priority = DMA_PRIORITY_LOW;
    if (HAL_DMA_Init(&hdma_dac_ch1) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(hdac,DMA_Handle1,hdma_dac_ch1);

  /* USER CODE BEGIN DAC1_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=30;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END DAC1_MspInit 1 */
  }

}

/**
* @brief DAC MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param hdac: DAC handle pointer
* @retval None
*/
void HAL_DAC_MspDeInit(DAC_HandleTypeDef* hdac)
{
  if(hdac->Instance==DAC1)
  {
  /* USER CODE BEGIN DAC1_MspDeInit 0 */

  /* USER CODE END DAC1_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_DAC1_CLK_DISABLE();

    /**DAC1 GPIO Configuration
    PA4     ------> DAC1_OUT1
    */
    HAL_GPIO_DeInit(GPIOA, GPIO_PIN_4);

    /* DAC1 DMA DeInit */
    HAL_DMA_DeInit(hdac->DMA_Handle1);
  /* USER CODE BEGIN DAC1_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=31;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END DAC1_MspDeInit 1 */
  }

}

/**
* @brief TIM_Base MSP Initialization
* This function configures the hardware resources used in this example
* @param htim_base: TIM_Base handle pointer
* @retval None
*/
void HAL_TIM_Base_MspInit(TIM_HandleTypeDef* htim_base)
{
  if(htim_base->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspInit 0 */

  /* USER CODE END TIM2_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_TIM2_CLK_ENABLE();
  /* USER CODE BEGIN TIM2_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=34;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM2_MspInit 1 */
  }
  else if(htim_base->Instance==TIM4)
  {
  /* USER CODE BEGIN TIM4_MspInit 0 */

  /* USER CODE END TIM4_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_TIM4_CLK_ENABLE();
  /* USER CODE BEGIN TIM4_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=37;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM4_MspInit 1 */
  }
  else if(htim_base->Instance==TIM5)
  {
  /* USER CODE BEGIN TIM5_MspInit 0 */

  /* USER CODE END TIM5_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_TIM5_CLK_ENABLE();
  /* USER CODE BEGIN TIM5_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=40;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM5_MspInit 1 */
  }
  else if(htim_base->Instance==TIM6)
  {
  /* USER CODE BEGIN TIM6_MspInit 0 */

  /* USER CODE END TIM6_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_TIM6_CLK_ENABLE();
  /* USER CODE BEGIN TIM6_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=43;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM6_MspInit 1 */
  }

}

void HAL_TIM_MspPostInit(TIM_HandleTypeDef* htim)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(htim->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspPostInit 0 */

  /* USER CODE END TIM2_MspPostInit 0 */
    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**TIM2 GPIO Configuration
    PA0     ------> TIM2_CH1
    PA1     ------> TIM2_CH2
    */
    GPIO_InitStruct.Pin = GPIO_PIN_0|GPIO_PIN_1;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM2;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /* USER CODE BEGIN TIM2_MspPostInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=35;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM2_MspPostInit 1 */
  }
  else if(htim->Instance==TIM4)
  {
  /* USER CODE BEGIN TIM4_MspPostInit 0 */

  /* USER CODE END TIM4_MspPostInit 0 */

    __HAL_RCC_GPIOB_CLK_ENABLE();
    /**TIM4 GPIO Configuration
    PB6     ------> TIM4_CH1
    */
    GPIO_InitStruct.Pin = GPIO_PIN_6;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF2_TIM4;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN TIM4_MspPostInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=38;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM4_MspPostInit 1 */
  }

}
/**
* @brief TIM_Base MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param htim_base: TIM_Base handle pointer
* @retval None
*/
void HAL_TIM_Base_MspDeInit(TIM_HandleTypeDef* htim_base)
{
  if(htim_base->Instance==TIM2)
  {
  /* USER CODE BEGIN TIM2_MspDeInit 0 */

  /* USER CODE END TIM2_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM2_CLK_DISABLE();
  /* USER CODE BEGIN TIM2_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=36;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM2_MspDeInit 1 */
  }
  else if(htim_base->Instance==TIM4)
  {
  /* USER CODE BEGIN TIM4_MspDeInit 0 */

  /* USER CODE END TIM4_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM4_CLK_DISABLE();
  /* USER CODE BEGIN TIM4_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=39;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM4_MspDeInit 1 */
  }
  else if(htim_base->Instance==TIM5)
  {
  /* USER CODE BEGIN TIM5_MspDeInit 0 */

  /* USER CODE END TIM5_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM5_CLK_DISABLE();
  /* USER CODE BEGIN TIM5_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=42;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM5_MspDeInit 1 */
  }
  else if(htim_base->Instance==TIM6)
  {
  /* USER CODE BEGIN TIM6_MspDeInit 0 */

  /* USER CODE END TIM6_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_TIM6_CLK_DISABLE();
  /* USER CODE BEGIN TIM6_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=45;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END TIM6_MspDeInit 1 */
  }

}

/**
* @brief UART MSP Initialization
* This function configures the hardware resources used in this example
* @param huart: UART handle pointer
* @retval None
*/
void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  if(huart->Instance==USART2)
  {
  /* USER CODE BEGIN USART2_MspInit 0 */

  /* USER CODE END USART2_MspInit 0 */
    /* Peripheral clock enable */
    __HAL_RCC_USART2_CLK_ENABLE();

    __HAL_RCC_GPIOA_CLK_ENABLE();
    /**USART2 GPIO Configuration
    PA2     ------> USART2_TX
    */
    GPIO_InitStruct.Pin = USART_TX_Pin;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
    GPIO_InitStruct.Pull = GPIO_PULLUP;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    GPIO_InitStruct.Alternate = GPIO_AF7_USART2;
    HAL_GPIO_Init(USART_TX_GPIO_Port, &GPIO_InitStruct);

    /* USART2 DMA Init */
    /* USART2_TX Init */
    hdma_usart2_tx.Instance = DMA1_Channel7;
    hdma_usart2_tx.Init.Request = DMA_REQUEST_2;
    hdma_usart2_tx.Init.Direction = DMA_MEMORY_TO_PERIPH;
    hdma_usart2_tx.Init.PeriphInc = DMA_PINC_DISABLE;
    hdma_usart2_tx.Init.MemInc = DMA_MINC_ENABLE;
    hdma_usart2_tx.Init.PeriphDataAlignment = DMA_PDATAALIGN_BYTE;
    hdma_usart2_tx.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    hdma_usart2_tx.Init.Mode = DMA_NORMAL;
    hdma_usart2_tx.Init.Priority = DMA_PRIORITY_LOW;
    if (HAL_DMA_Init(&hdma_usart2_tx) != HAL_OK)
    {
      Error_Handler();
    }

    __HAL_LINKDMA(huart,hdmatx,hdma_usart2_tx);

    /* USART2 interrupt Init */
    HAL_NVIC_SetPriority(USART2_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART2_IRQn);
  /* USER CODE BEGIN USART2_MspInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=28;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END USART2_MspInit 1 */
  }

}

/**
* @brief UART MSP De-Initialization
* This function freeze the hardware resources used in this example
* @param huart: UART handle pointer
* @retval None
*/
void HAL_UART_MspDeInit(UART_HandleTypeDef* huart)
{
  if(huart->Instance==USART2)
  {
  /* USER CODE BEGIN USART2_MspDeInit 0 */

  /* USER CODE END USART2_MspDeInit 0 */
    /* Peripheral clock disable */
    __HAL_RCC_USART2_CLK_DISABLE();

    /**USART2 GPIO Configuration
    PA2     ------> USART2_TX
    */
    HAL_GPIO_DeInit(USART_TX_GPIO_Port, USART_TX_Pin);

    /* USART2 DMA DeInit */
    HAL_DMA_DeInit(huart->hdmatx);

    /* USART2 interrupt DeInit */
    HAL_NVIC_DisableIRQ(USART2_IRQn);
  /* USER CODE BEGIN USART2_MspDeInit 1 */
{
        {
                        extern TIM_HandleTypeDef htim5 ;
                {
                                    __auto_type prim  = __get_PRIMASK();
            __disable_irq();
                                    glog_ts[glog_count]=htim5.Instance->CNT;
                        glog_msg[glog_count]=29;
                        (glog_count)++;
            if ( (2048)<=(glog_count) ) {
                                                                glog_count=0;
}
            if ( !(prim) ) {
                                                __enable_irq();
}
}
}
}
/* USER CODE END USART2_MspDeInit 1 */
  }

}

/* USER CODE BEGIN 1 */

/* USER CODE END 1 */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
