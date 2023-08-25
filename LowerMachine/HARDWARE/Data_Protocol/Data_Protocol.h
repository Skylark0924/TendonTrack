#ifndef __DATA_PRTOCOL_H
#define __DATA_PRTOCOL_H
 
 
extern unsigned char DataSend_Buffer[42];	   	//发送帧数据缓存区
extern unsigned char DataReceive_Buff[100];		//接收数据缓存
extern int	receiveBuffer_cnt;

void dataSendBuffer_GetData(float Data,unsigned char Channel);    // 写通道数据至 待发送帧数据缓存区
unsigned char dataProtocol_SendPack(unsigned char Channel_Number);  // 发送帧数据生成函数 

float dataProtocol_Unpack();
#endif 
