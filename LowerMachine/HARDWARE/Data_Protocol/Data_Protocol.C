#include "Data_Protocol.h"

unsigned char DataSend_Buffer[42] = {0};	   //���ڷ��ͻ�����
unsigned char DataReceive_Buff[100] = {0};
int	receiveBuffer_cnt = 0;

//����˵�����������ȸ�������ת��4�ֽ����ݲ�����ָ����ַ 
//����˵�����û�����ֱ�Ӳ����˺��� 
//target:Ŀ�굥��������
//buf:��д������
//beg:ָ��������ڼ���Ԫ�ؿ�ʼд��
//�����޷��� 
void Float2Byte(float *target,unsigned char *buf,unsigned char beg)
{
    unsigned char *point;
    point = (unsigned char*)target;	  //�õ�float�ĵ�ַ
    buf[beg]   = point[0];
    buf[beg+1] = point[1];
    buf[beg+2] = point[2];
    buf[beg+3] = point[3];
}

float Byte2Float(unsigned char *buf,unsigned char beg)
{
    float output;
		float *point;
    point = (float *)(buf+beg);	  //�õ�float�ĵ�ַ
    output   = point[0];
		return output;
}
 
//����˵������������ͨ���ĵ����ȸ�������д�뷢�ͻ�����
//Data��ͨ������
//Channel��ѡ��ͨ����1-10��
//�����޷��� 
void dataSendBuffer_GetData(float Data,unsigned char Channel)
{
	if ( (Channel > 10) || (Channel == 0) ) return;  //ͨ����������10�����0��ֱ����������ִ�к���
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


//����˵�������� DataScopeV1.0 ����ȷʶ���֡��ʽ
//Channel_Number����Ҫ���͵�ͨ������
//���ط��ͻ��������ݸ���
//����0��ʾ֡��ʽ����ʧ�� 
unsigned char dataProtocol_SendPack(unsigned char Channel_Number)
{
	if ( (Channel_Number > 10) || (Channel_Number == 0) ) { return 0; }  //ͨ����������10�����0��ֱ����������ִ�к���
  else
  {	
	 DataSend_Buffer[0] = '$';  //֡ͷ
		
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
	if(DataReceive_Buff[head_idx]!='$')	//˵������������
	{
		int i;
		for(i=head_idx;i<100-4;i++)						//Ѱ��֡ͷ
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