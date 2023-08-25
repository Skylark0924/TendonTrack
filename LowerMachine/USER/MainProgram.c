#include "sys.h"

u8 Flag_Stop=1,delay_50,delay_20,delay_flag,motor_flag;         //ֹͣ��־λ 50ms��׼��ʾ��־λ
int Encoder,Position_Zero=10000;            //���������������
int Moto;                                   //���PWM���� Ӧ��Motor�� ��Moto�¾�	
int Voltage;                                //��ص�ѹ������صı���
float Angle_Balance;                        //��λ�ƴ���������
float Balance_KP=400,Balance_KD=400,Position_KP=20,Position_KD=300;  //PIDϵ��
float Menu=1,Amplitude1=5,Amplitude2=20,Amplitude3=1,Amplitude4=10; //PID������ز���
float Rec_TEST=0;

int main(void)
{ 
	Stm32_Clock_Init(9);            //=====ϵͳʱ������
	delay_init(72);                 //=====��ʱ��ʼ��
	JTAG_Set(JTAG_SWD_DISABLE);     //=====�ر�JTAG�ӿ�
	JTAG_Set(SWD_ENABLE);           //=====��SWD�ӿ� �������������SWD�ӿڵ���
	delay_ms(1000);                 //=====��ʱ�������ȴ�ϵͳ�ȶ�
	//delay_ms(1000);                 //=====��ʱ�������ȴ�ϵͳ�ȶ� ��2s
	LED_Init();                     //=====��ʼ���� LED ���ӵ�Ӳ���ӿ�
	EXTI_Init();                    //=====������ʼ��(�ⲿ�жϵ���ʽ)
	//OLED_Init();                    //=====OLED��ʼ��
	uart_init(72,128000);           //=====��ʼ������1
  Motor_PWM_Init(7199,0);   			//=====��ʼ��PWM 10KHZ������������� 
	//Encoder_Init_TIM4();            //=====��ʼ����������TIM2�ı������ӿ�ģʽ�� 
	//Angle_Adc_Init();               //=====��λ�ƴ�����ģ�����ɼ���ʼ��
	//Baterry_Adc_Init();             //=====��ص�ѹģ�����ɼ���ʼ��
	Timer1_Init(49,7199);           //=====��ʱ�жϳ�ʼ�� 
	motor_flag=0;
	while(1)
		{      
				dataTxRx();	            //===����λ��ͨѶ
				delay_flag=1;	            //===50ms�жϾ�׼��ʱ��־λ
				oled_show();              //===��ʾ����	  	
				while(delay_flag);        //===50ms�жϾ�׼��ʱ  ��Ҫ�ǲ�����ʾ��λ����Ҫ�ϸ��50ms��������   							
		} 
}
