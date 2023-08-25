#ifndef __MOTOR_H
#define __MOTOR_H
#include <sys.h>	 

#define PWMA   TIM3->CCR4    //倒立摆仅使用一路驱动 使用的A路
#define AIN2   PBout(12)     //倒立摆仅使用一路驱动 使用的A路
#define AIN1   PBout(13)     //倒立摆仅使用一路驱动 使用的A路
#define BIN1   PBout(14)     //预留 备用
#define BIN2   PBout(15)     //预留 备用
#define PWMB   TIM3->CCR3    //预留 备用
void Motor_PWM_Init(u16 arr,u16 psc);
void Motor_Init(void);
#endif
