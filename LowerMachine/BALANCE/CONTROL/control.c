#include "control.h"		

int Balance_Pwm,Position_Pwm;
u8 Flag_Target,Position_Target;
/**************************************************************************
�������ܣ����еĿ��ƴ��붼��������
          TIM1���Ƶ�5ms��ʱ�ж� 
**************************************************************************/
int TIM1_UP_IRQHandler(void)  
{    
	if(TIM1->SR&0X0001)//5ms��ʱ�ж�
	{   
		   TIM1->SR&=~(1<<0);                                       //===�����ʱ��1�жϱ�־λ	                     
	     if(delay_flag==1)
			 {
				 if(++delay_20==4)	 delay_20=0,delay_flag=0;          //===���������ṩ50ms�ľ�׼��ʱ
			 }		
    	//Encoder=Read_Encoder(4);             	                   //===���±�����λ����Ϣ
      //Angle_Balance=Get_Adc_Average(3,15);                     //===������̬	
     	//Balance_Pwm =balance(Angle_Balance);                                          //===�Ƕ�PD����	
	    //if(++Position_Target>4)	Position_Pwm=Position(Encoder),Position_Target=0;     //===λ��PD���� 25ms����һ��λ�ÿ���
      //Moto=Balance_Pwm-Position_Pwm;        //===����������PWM
			
			Moto=2000;
		  Xianfu_Pwm();                         //===PWM�޷� ����ռ�ձ�100%������ϵͳ���ȶ�����
		  if(Turn_Off()==0)              //===��ѹ����ǹ��󱣻�	Voltage
			
			if(Rec_TEST==1){		//Rec_TEST
				Set_Pwm(Moto);                        //===��ֵ��PWM�Ĵ���
				Led_Flash(50);                       //===LED��˸ָʾϵͳ�������� 
			}
			else if(Rec_TEST==2){
				//Moto=800;
				Set_Pwm(-Moto);
				Led_Flash(50);
			}
			else{
				Set_Pwm(0);
				Led_Flash(0);
			}
			
			if(Flag_Stop==1){
				Led_Flash(0);
			}
			else{
				Led_Flash(100);
			}
			//Key_Click();
	    //Voltage=Get_battery_volt();           //===��ȡ��ص�ѹ	      
			//Key();                                //===ɨ�谴���仯
	}       	
	 return 0;	  
} 

/**************************************************************************
�������ܣ����PD����
��ڲ������Ƕ�
����  ֵ����ǿ���PWM
**************************************************************************/
int balance(float Angle)
{  
   float Bias;                       //���ƫ��
	 static float Last_Bias,D_Bias;    //PID��ر���
	 int balance;                      //PWM����ֵ 
	 Bias=Angle-ZHONGZHI;              //���ƽ��ĽǶ���ֵ �ͻ�е���
	 D_Bias=Bias-Last_Bias;            //���ƫ���΢�� ����΢�ֿ���
	 balance=-Balance_KP*Bias-D_Bias*Balance_KD;   //===������ǿ��Ƶĵ��PWM  PD����
   Last_Bias=Bias;                   //������һ�ε�ƫ��
	 return balance;
}

/**************************************************************************
�������ܣ�λ��PD���� 
��ڲ�����������
����  ֵ��λ�ÿ���PWM
**************************************************************************/
int Position(int Encoder)
{  
   static float Position_PWM,Last_Position,Position_Bias,Position_Differential;
	 static float Position_Least;
  	Position_Least =Encoder-Position_Zero;             //===
    Position_Bias *=0.8;		   
    Position_Bias += Position_Least*0.2;	             //===һ�׵�ͨ�˲���  
	  Position_Differential=Position_Bias-Last_Position;
	  Last_Position=Position_Bias;
		Position_PWM=Position_Bias*Position_KP+Position_Differential*Position_KD; //===�ٶȿ���	
	  return Position_PWM;
}

/**************************************************************************
�������ܣ���ֵ��PWM�Ĵ���
��ڲ�����PWM
����  ֵ����
**************************************************************************/
void Set_Pwm(int moto)
{
    	if(moto<0)			AIN2=1,			AIN1=0;
			else 	          AIN2=0,			AIN1=1;
			PWMA=myabs(moto);
}

/**************************************************************************
�������ܣ�����PWM��ֵ 
��ڲ�������
����  ֵ����
**************************************************************************/
void Xianfu_Pwm(void)
{	
	  int Amplitude=6900;    //===PWM������7200 ������6900
	  if(Moto<-Amplitude) Moto=-Amplitude;	
		if(Moto>Amplitude)  Moto=Amplitude;		
}

/**************************************************************************
�������ܣ������޸�С������״̬  ���ưڸ˵�λ��
��ڲ�������
����  ֵ����
**************************************************************************/
void Key_Click(void)
{	
	static int tmp,flag,count;
	tmp=click_N_Double(100); 
	
	if(tmp==1)flag=1;//++
  if(tmp==2)flag=2;//--
	
	if(flag==1) //�˶�
	{
		motor_flag=1;
	}	
		if(flag==2) //ֹͣ
	{
		motor_flag=2;
	}

//	if(flag==1) //�ڸ�˳ʱ���˶�
//	{
//		Position_Zero+=4;
//		count+=4;	
//		if(count==Position) 	flag=0,count=0;
//	}	
//		if(flag==2) //�ڸ���ʱ���˶�
//	{
//		Position_Zero-=4;
//		count+=4;	
//		if(count==Position) 	flag=0,count=0;
//	}
}

/**************************************************************************
�������ܣ������޸�С������״̬  ���ưڸ˵�λ��
��ڲ�������
����  ֵ����
**************************************************************************/
void Key(void)
{	
	int Position=1040; //Ŀ��λ�� ���ԭʼλ����10000  תһȦ��1040 �ͱ����������йأ�Ĭ���ǵ��תһȦ�����1040��������
	static int tmp,flag,count;
	tmp=click_N_Double(100); 
	
	if(tmp==1)flag=1;//++
  if(tmp==2)flag=2;//--
	
	if(flag==1) //�ڸ�˳ʱ���˶�
	{
		Position_Zero+=4;
		count+=4;	
		if(count==Position) 	flag=0,count=0;
	}	
		if(flag==2) //�ڸ���ʱ���˶�
	{
		Position_Zero-=4;
		count+=4;	
		if(count==Position) 	flag=0,count=0;
	}
}

/**************************************************************************
�������ܣ��쳣�رյ��
��ڲ�������ѹ
����  ֵ��1���쳣  0������
**************************************************************************/
u8 Turn_Off(void)	//int voltage
{
	    u8 temp; 
	    static u8 count;
			if(1==Flag_Stop) //��ص�ѹ���ͣ��رյ��
			{	      
				Flag_Stop=1;				
				temp=1;                                            
				AIN1=0;                                            
				AIN2=0;
      }
			else
				temp=0;
			
			
//			if(Angle_Balance<(ZHONGZHI-800)||Angle_Balance>(ZHONGZHI+800))count++;else count=0;//||voltage<700
//			if(count==120)
//			{
//				count=0;
//				Flag_Stop=1;
//			}	
      return temp;			
}
/**************************************************************************
�������ܣ�����ֵ����
��ڲ�����int
����  ֵ��unsigned int
**************************************************************************/
int myabs(int a)
{ 		   
	  int temp;
		if(a<0)  temp=-a;  
	  else temp=a;
	  return temp;
}
