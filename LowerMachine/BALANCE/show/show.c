#include "show.h"
//#include <string.h>/* memset */

unsigned char i,temp;          //��������
unsigned char Send_Count; //������Ҫ���͵����ݸ���
float Vol;
/**************************************************************************
�������ܣ�OLED��ʾ
��ڲ�������
����  ֵ����
**************************************************************************/
void oled_show(void)
{
		//=============��1����ʾ�Ƕ�PD����P����Position_KP=======================//	
		                      OLED_ShowString(00,00,"B-KP");                   
		                      OLED_ShowNumber(40,00,Balance_KP,3,12);
	                        OLED_ShowString(57,00,"."),  
	                        OLED_ShowNumber(61,00,(int)(Balance_KP*10)%10,1,12);
	
	                        OLED_ShowString(95,00,"A:");	  
	                        OLED_ShowNumber(108,00,Amplitude1,2,12);
		//=============��2����ʾ�Ƕ�PD����D����=======================//	
		                      OLED_ShowString(00,10,"B-KD");
		                      OLED_ShowNumber(40,10,Balance_KD,3,12);
	                        OLED_ShowString(57,10,"."),  
	                        OLED_ShowNumber(61,10,(int)(Balance_KD*10)%10,1,12);
	
	                        OLED_ShowString(95,10,"A:");	  
	                        OLED_ShowNumber(108,10,Amplitude2,2,12);
		//=============��3����ʾ������Position_KP=======================//	
		                      OLED_ShowString(00,20,"P-KP");
		                      OLED_ShowNumber(40,20,Position_KP,3,12);
	                        OLED_ShowString(57,20,"."),  
	                        OLED_ShowNumber(61,20,(int)(Position_KP*10)%10,1,12);
													
												  OLED_ShowString(95,20,"A:");	  
	                        OLED_ShowNumber(108,20,Amplitude3,2,12);
		//=============��4����ʾ������1=======================//	
		                      OLED_ShowString(00,30,"P-KD");
		                      OLED_ShowNumber(40,30,Position_KD,3,12);
	                        OLED_ShowString(57,30,"."),  
	                        OLED_ShowNumber(61,30,(int)(Position_KD*10)%10,1,12);
													
													OLED_ShowString(95,30,"A:");	  
	                        OLED_ShowNumber(108,30,Amplitude4,2,12);
		//======���ǹ����˵� ѡ����Ҫ���ڵ�PD����											
		  if(Menu==1)
	   	{
			 OLED_ShowChar(75,00,'Y',12,1);   
			 OLED_ShowChar(75,10,'N',12,1);
			 OLED_ShowChar(75,20,'N',12,1);
			 OLED_ShowChar(75,30,'N',12,1);
	  	}
		  else	if(Menu==2)
	   	{
			 OLED_ShowChar(75,00,'N',12,1);
			 OLED_ShowChar(75,10,'Y',12,1);
			 OLED_ShowChar(75,20,'N',12,1);
			 OLED_ShowChar(75,30,'N',12,1);
			}		
      else if(Menu==3)
	   	{			
			 OLED_ShowChar(75,00,'N',12,1);
			 OLED_ShowChar(75,10,'N',12,1);
			 OLED_ShowChar(75,20,'Y',12,1);
			 OLED_ShowChar(75,30,'N',12,1);
			}		
      else if(Menu==4)
	   	{				
			 OLED_ShowChar(75,00,'N',12,1);
			 OLED_ShowChar(75,10,'N',12,1);
			 OLED_ShowChar(75,20,'N',12,1);
			 OLED_ShowChar(75,30,'Y',12,1);
	 	  } 
	//=============��������ʾ��ѹ��Ŀ��λ��=======================//			
			OLED_ShowString(80,40,"T:");	  
			OLED_ShowNumber(95,40,Position_Zero,5,12) ; 
			                    OLED_ShowString(00,40,"VOL:");
		                      OLED_ShowString(41,40,".");
		                      OLED_ShowString(63,40,"R");
		                      OLED_ShowNumber(28,40,Rec_TEST,2,12);
		                      OLED_ShowNumber(51,40,Flag_Stop,2,12);
		 if(Voltage%100<10) 	OLED_ShowNumber(45,40,0,2,12);
		//=============��������ʾ��λ�ƴ������ͱ���������=======================//
		OLED_ShowString(80,50,"P:");    OLED_ShowNumber(95,50,Encoder,5,12); 
		OLED_ShowString(0,50,"ADC:");  OLED_ShowNumber(30,50,Angle_Balance,4,12);
		//=============ˢ��=======================//
		OLED_Refresh_Gram();	
	}

/**************************************************************************
�������ܣ�����ʾ��������λ���������� �ر���ʾ��
��ڲ�������
����  ֵ����
**************************************************************************/
void dataTxRx(void)
{   

		dataSendBuffer_GetData( Angle_Balance, 1 );      
		dataSendBuffer_GetData( Encoder, 2 );         
		dataSendBuffer_GetData( Moto, 3 );              
//		dataSendBuffer_GetData( 0 , 4 );   
//		dataSendBuffer_GetData(0, 5 ); //����Ҫ��ʾ�������滻0������
//		dataSendBuffer_GetData(0 , 6 );//����Ҫ��ʾ�������滻0������
//		dataSendBuffer_GetData(0, 7 );
//		dataSendBuffer_GetData( 0, 8 ); 
//		dataSendBuffer_GetData(0, 9 );  
//		dataSendBuffer_GetData( 0 , 10);
		Send_Count = dataProtocol_SendPack(3);
		for( i = 0 ; i < Send_Count; i++) 
		{
			while((USART1->SR&0X40)==0);  
			USART1->DR = DataSend_Buffer[i]; 
		}
}

/**************************ʵ�ֺ���**********************************************
*��    ��:		usart1�����ж�
*********************************************************************************/
void USART1_IRQHandler(void)
{	
	if(USART1->SR&(1<<5))//���յ�����
	{	  
		static	int uart_receive=0;//����������ر���
		float tmp;
		uart_receive=USART1->DR; 
		if(receiveBuffer_cnt>=100){
			receiveBuffer_cnt=0;
			memset(DataReceive_Buff,0,sizeof(DataReceive_Buff));
		}
		
		DataReceive_Buff[receiveBuffer_cnt]=uart_receive;
		receiveBuffer_cnt++;

		tmp=dataProtocol_Unpack();
		if(tmp!=-1) Rec_TEST=tmp;
	
	}  											 
} 
