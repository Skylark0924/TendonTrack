#ifndef __DATA_PRTOCOL_H
#define __DATA_PRTOCOL_H
 
 
extern unsigned char DataSend_Buffer[42];	   	//����֡���ݻ�����
extern unsigned char DataReceive_Buff[100];		//�������ݻ���
extern int	receiveBuffer_cnt;

void dataSendBuffer_GetData(float Data,unsigned char Channel);    // дͨ�������� ������֡���ݻ�����
unsigned char dataProtocol_SendPack(unsigned char Channel_Number);  // ����֡�������ɺ��� 

float dataProtocol_Unpack();
#endif 
