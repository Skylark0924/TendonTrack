#include "Data_Protocol.h"

unsigned char DataSend_Buffer[42] = {0};	   //串口发送缓冲区
unsigned char DataReceive_Buff[100] = {0};
int	receiveBuffer_cnt = 0;

//函数说明：将单精度浮点数据转成4字节数据并存入指定地址 
//附加说明：用户无需直接操作此函数 
//target:目标单精度数据
//buf:待写入数组
//beg:指定从数组第几个元素开始写入
//函数无返回 
void Float2Byte(float *target,unsigned char *buf,unsigned char beg)
{
    unsigned char *point;
    point = (unsigned char*)target;	  //得到float的地址
    buf[beg]   = point[0];
    buf[beg+1] = point[1];
    buf[beg+2] = point[2];
    buf[beg+3] = point[3];
}

float Byte2Float(unsigned char *buf,unsigned char beg)
{
    float output;
		float *point;
    point = (float *)(buf+beg);	  //得到float的地址
    output   = point[0];
		return output;
}
 
//函数说明：将待发送通道的单精度浮点数据写入发送缓冲区
//Data：通道数据
//Channel：选择通道（1-10）
//函数无返回 
void dataSendBuffer_GetData(float Data,unsigned char Channel)
{
	if ( (Channel > 10) || (Channel == 0) ) return;  //通道个数大于10或等于0，直接跳出，不执行函数
  else
  {
     switch (Channel)
		{
      case 1:  Float2Byte(&Data,DataSend_Buffer,1); break;
      case 2:  Float2Byte(&Data,DataSend_Buffer,5); break;
		  case 3:  Float2Byte(&Data,DataSend_Buffer,9); break;
		  case 4:  Float2Byte(&Data,DataSend_Buffer,13); break;
		  case 5:  Float2Byte(&Data,DataSend_Buffer,17); break;
		  case 6:  Float2Byte(&Data,DataSend_Buffer,21); break;
		  case 7:  Float2Byte(&Data,DataSend_Buffer,25); break;
		  case 8:  Float2Byte(&Data,DataSend_Buffer,29); break;
		  case 9:  Float2Byte(&Data,DataSend_Buffer,33); break;
		  case 10: Float2Byte(&Data,DataSend_Buffer,37); break;
		}
  }	 
}


//函数说明：生成 DataScopeV1.0 能正确识别的帧格式
//Channel_Number，需要发送的通道个数
//返回发送缓冲区数据个数
//返回0表示帧格式生成失败 
unsigned char dataProtocol_SendPack(unsigned char Channel_Number)
{
	if ( (Channel_Number > 10) || (Channel_Number == 0) ) { return 0; }  //通道个数大于10或等于0，直接跳出，不执行函数
  else
  {	
	 DataSend_Buffer[0] = '$';  //帧头
		
	 switch(Channel_Number)   
   { 
		 case 1:   DataSend_Buffer[5]  =  5; return  6;  
		 case 2:   DataSend_Buffer[9]  =  9; return 10;
		 case 3:   DataSend_Buffer[13] = 13; return 14; 
		 case 4:   DataSend_Buffer[17] = 17; return 18;
		 case 5:   DataSend_Buffer[21] = 21; return 22;  
		 case 6:   DataSend_Buffer[25] = 25; return 26;
		 case 7:   DataSend_Buffer[29] = 29; return 30; 
		 case 8:   DataSend_Buffer[33] = 33; return 34; 
		 case 9:   DataSend_Buffer[37] = 37; return 38;
     case 10:  DataSend_Buffer[41] = 41; return 42; 
   }	 
  }
	return 0;
}


float dataProtocol_Unpack()
{
	static int head_idx	= 0;
	if(DataReceive_Buff[head_idx]!='$')	//说明缓冲区清零
	{
		int i;
		for(i=head_idx;i<100-4;i++)						//寻找帧头
		{						
			if(DataReceive_Buff[i]=='$')
			{
				head_idx=i;
				break;
			}
		}
	}
	
	if(DataReceive_Buff[head_idx]!='$') {head_idx=0;return -1;}
	
	if(DataReceive_Buff[head_idx+5]!='&') return -1;
	else{
		float tmp=Byte2Float(DataReceive_Buff,head_idx+1);
		head_idx+=6;
		return tmp;
	}
}
