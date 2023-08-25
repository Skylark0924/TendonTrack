#ifndef __CONTROL_H
#define __CONTROL_H
#include "sys.h"

#define PI 3.14159265
#define ZHONGZHI 3110
extern	int Balance_Pwm,Velocity_Pwm;
int TIM1_UP_IRQHandler(void);
int balance(float angle);
int Position(int Encoder);
void Set_Pwm(int moto);
void Key_Click(void);
void Key(void);
void Xianfu_Pwm(void);
u8 Turn_Off(void);//int voltage
int myabs(int a);
#endif
