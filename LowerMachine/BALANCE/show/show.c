#include "show.h"
//#include <string.h>/* memset */

unsigned char i,temp;          //计数变量
unsigned char Send_Count; //串口需要发送的数据个数
float Vol;
/**************************************************************************
函数功能：OLED显示
入口参数：无
返回  值：无
**************************************************************************/
void oled_show(void)
{
		//=============第1行显示角度PD控制P参数Position_KP=======================//	
		                      OLED_ShowString(00,00,"B-KP");                   
		                      OLED_ShowNumber(40,00,Balance_KP,3,12);
	                        OLED_ShowString(57,00,"."),  
	                        OLED_ShowNumber(61,00,(int)(Balance_KP*10)%10,1,12);
	
	                        OLED_ShowString(95,00,"A:");	  
	                        OLED_ShowNumber(108,00,Amplitude1,2,12);
		//=============第2行显示角度PD控制D参数=======================//	
		                      OLED_ShowString(00,10,"B-KD");
		                      OLED_ShowNumber(40,10,Balance_KD,3,12);
	                        OLED_ShowString(57,10,"."),  
	                        OLED_ShowNumber(61,10,(int)(Balance_KD*10)%10,1,12);
	
	                        OLED_ShowString(95,10,"A:");	  
	                        OLED_ShowNumber(108,10,Amplitude2,2,12);
		//=============第3行显示编码器Position_KP=======================//	
		                      OLED_ShowString(00,20,"P-KP");
		                      OLED_ShowNumber(40,20,Position_KP,3,12);
	                        OLED_ShowString(57,20,"."),  
	                        OLED_ShowNumber(61,20,(int)(Position_KP*10)%10,1,12);
													
												  OLED_ShowString(95,20,"A:");	  
	                        OLED_ShowNumber(108,20,Amplitude3,2,12);
		//=============第4行显示编码器1=======================//	
		                      OLED_ShowString(00,30,"P-KD");
		                      OLED_ShowNumber(40,30,Position_KD,3,12);
	                        OLED_ShowString(57,30,"."),  
	                        OLED_ShowNumber(61,30,(int)(Position_KD*10)%10,1,12);
													
													OLED_ShowString(95,30,"A:");	  
	                        OLED_ShowNumber(108,30,Amplitude4,2,12);
		//======这是滚动菜单 选择需要调节的PD参数											
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
	//=============第五行显示电压和目标位置=======================//			
			OLED_ShowString(80,40,"T:");	  
			OLED_ShowNumber(95,40,Position_Zero,5,12) ; 
			                    OLED_ShowString(00,40,"VOL:");
		                      OLED_ShowString(41,40,".");
		                      OLED_ShowString(63,40,"R");
		                      OLED_ShowNumber(28,40,Rec_TEST,2,12);
		                      OLED_ShowNumber(51,40,Flag_Stop,2,12);
		 if(Voltage%100<10) 	OLED_ShowNumber(45,40,0,2,12);
		//=============第六行显示角位移传感器和编码器数据=======================//
		OLED_ShowString(80,50,"P:");    OLED_ShowNumber(95,50,Encoder,5,12); 
		OLED_ShowString(0,50,"ADC:");  OLED_ShowNumber(30,50,Angle_Balance,4,12);
		//=============刷新=======================//
		OLED_Refresh_Gram();	
	}

/**************************************************************************
函数功能：虚拟示波器往上位机发送数据 关闭显示屏
入口参数：无
返回  值：无
**************************************************************************/
void dataTxRx(void)
{   

		dataSendBuffer_GetData( Angle_Balance, 1 );      
		dataSendBuffer_GetData( Encoder, 2 );         
		dataSendBuffer_GetData( Moto, 3 );              
//		dataSendBuffer_GetData( 0 , 4 );   
//		dataSendBuffer_GetData(0, 5 ); //用您要显示的数据替换0就行了
//		dataSendBuffer_GetData(0 , 6 );//用您要显示的数据替换0就行了
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

/**************************实现函数**********************************************
*功    能:		usart1接受中断
*********************************************************************************/
void USART1_IRQHandler(void)
{	
	if(USART1->SR&(1<<5))//接收到数据
	{	  
		static	int uart_receive=0;//蓝牙接收相关变量
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
