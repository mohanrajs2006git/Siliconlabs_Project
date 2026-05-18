#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

// $[USART0]
// [USART0]$

// $[UART1]
// [UART1]$

// $[ULP_UART]
// ULP_UART TX on ULP_GPIO_11/GPIO_75
#ifndef ULP_UART_TX_PORT                        
#define ULP_UART_TX_PORT                         ULP
#endif
#ifndef ULP_UART_TX_PIN                         
#define ULP_UART_TX_PIN                          11
#endif
#ifndef ULP_UART_TX_LOC                         
#define ULP_UART_TX_LOC                          1
#endif

// ULP_UART RX on ULP_GPIO_9/GPIO_73
#ifndef ULP_UART_RX_PORT                        
#define ULP_UART_RX_PORT                         ULP
#endif
#ifndef ULP_UART_RX_PIN                         
#define ULP_UART_RX_PIN                          9
#endif
#ifndef ULP_UART_RX_LOC                         
#define ULP_UART_RX_LOC                          3
#endif

// [ULP_UART]$

// $[I2C0]
// [I2C0]$

// $[I2C1]
// [I2C1]$

// $[ULP_I2C]
// [ULP_I2C]$

// $[SSI_MASTER]
// [SSI_MASTER]$

// $[SSI_SLAVE]
// [SSI_SLAVE]$

// $[ULP_SSI]
// [ULP_SSI]$

// $[GSPI_MASTER]
// [GSPI_MASTER]$

// $[I2S0]
// I2S0 SCLK on GPIO_46
#ifndef I2S0_SCLK_PORT                          
#define I2S0_SCLK_PORT                           HP
#endif
#ifndef I2S0_SCLK_PIN                           
#define I2S0_SCLK_PIN                            46
#endif
#ifndef I2S0_SCLK_LOC                           
#define I2S0_SCLK_LOC                            2
#endif

// I2S0 WSCLK on GPIO_47
#ifndef I2S0_WSCLK_PORT                         
#define I2S0_WSCLK_PORT                          HP
#endif
#ifndef I2S0_WSCLK_PIN                          
#define I2S0_WSCLK_PIN                           47
#endif
#ifndef I2S0_WSCLK_LOC                          
#define I2S0_WSCLK_LOC                           6
#endif

// I2S0 DOUT0 on GPIO_28
#ifndef I2S0_DOUT0_PORT                         
#define I2S0_DOUT0_PORT                          HP
#endif
#ifndef I2S0_DOUT0_PIN                          
#define I2S0_DOUT0_PIN                           28
#endif
#ifndef I2S0_DOUT0_LOC                          
#define I2S0_DOUT0_LOC                           9
#endif

// I2S0 DIN0 on GPIO_48
#ifndef I2S0_DIN0_PORT                          
#define I2S0_DIN0_PORT                           HP
#endif
#ifndef I2S0_DIN0_PIN                           
#define I2S0_DIN0_PIN                            48
#endif
#ifndef I2S0_DIN0_LOC                           
#define I2S0_DIN0_LOC                            14
#endif

// [I2S0]$

// $[ULP_I2S]
// [ULP_I2S]$

// $[SCT]
// [SCT]$

// $[SIO]
// [SIO]$

// $[PWM]
// [PWM]$

// $[PWM_CH0]
// [PWM_CH0]$

// $[PWM_CH1]
// [PWM_CH1]$

// $[PWM_CH2]
// [PWM_CH2]$

// $[PWM_CH3]
// [PWM_CH3]$

// $[ADC_CH1]
// [ADC_CH1]$

// $[ADC_CH2]
// [ADC_CH2]$

// $[ADC_CH3]
// [ADC_CH3]$

// $[ADC_CH4]
// [ADC_CH4]$

// $[ADC_CH5]
// [ADC_CH5]$

// $[ADC_CH6]
// [ADC_CH6]$

// $[ADC_CH7]
// [ADC_CH7]$

// $[ADC_CH8]
// [ADC_CH8]$

// $[ADC_CH9]
// [ADC_CH9]$

// $[ADC_CH10]
// [ADC_CH10]$

// $[ADC_CH11]
// [ADC_CH11]$

// $[ADC_CH12]
// [ADC_CH12]$

// $[ADC_CH13]
// [ADC_CH13]$

// $[ADC_CH14]
// [ADC_CH14]$

// $[ADC_CH15]
// [ADC_CH15]$

// $[ADC_CH16]
// [ADC_CH16]$

// $[ADC_CH17]
// [ADC_CH17]$

// $[ADC_CH18]
// [ADC_CH18]$

// $[ADC_CH19]
// [ADC_CH19]$

// $[COMP1]
// [COMP1]$

// $[COMP2]
// [COMP2]$

// $[DAC0]
// [DAC0]$

// $[DAC1]
// [DAC1]$

// $[SYSRTC]
// [SYSRTC]$

// $[UULP_VBAT_GPIO]
// [UULP_VBAT_GPIO]$

// $[GPIO]
// [GPIO]$

// $[QEI]
// [QEI]$

// $[HSPI_SECONDARY]
// [HSPI_SECONDARY]$

// $[CUSTOM_PIN_NAME]
#ifndef _PORT                                   
#define _PORT                                    HP
#endif
#ifndef _PIN                                    
#define _PIN                                     6
#endif

// [CUSTOM_PIN_NAME]$

#endif // PIN_CONFIG_H
// $[SDC_CH1]
// [SDC_CH1]$

// $[SDC_CH2]
// [SDC_CH2]$

// $[SDC_CH3]
// [SDC_CH3]$

// $[SDC_CH4]
// [SDC_CH4]$

// $[OPAMP1]
// [OPAMP1]$

// $[OPAMP2]
// [OPAMP2]$

// $[OPAMP3]
// [OPAMP3]$

