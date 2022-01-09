#ifndef COMMAND_H
#define COMMAND_H

#include "para.h"
#include <math.h>
#include <string>
#include <vector>
#include "command.h"
#include <iostream>
#include <assert.h>


namespace Command {

void							SwitchLeBe(UINT32 &a);
void							SwitchLeBe(UINT16 &a);
void							SwitchLeBe(UINT64 &a);
void							SwitchLeBe(NetQueryReceiveCMD &cmd);

void                            DecMacToHex(UINT64 nDecMac, UINT8 *pHexMac);
UINT64                          GetMacDec(const UINT8 *pMac);
std::string                     GetMacStr(const UINT8 *pMac);

std::string                     GetCalibPrefix(int nCalibId);

//QString                         GetSrcFile(int nBDMId, QString qstrSavePath, QString qstrPrefix = "");
//QString                         GetResolveFile(int nBDMId, QString qstrSavePath, QString qstrPrefix = "");
//
//QString                         GetSrcFile(const UINT8 nMAC[6], QString qstrSavePath, QString qstrPrefix = "");
//QString                         GetResolveFile(const UINT8 nMAC[6], QString qstrSavePath, QString qstrPrefix = "");

std::string                     GetSimplified(double fNumber, int nDecimals);

UINT8						    GetVerifyValue(UINT8 nValue1, UINT8 nValue2, UINT8 nValue3, UINT8 nValue4, UINT8 nValue5);
ErrorType					    ReturnType(const Instruction &cmd);
bool							IsReturnSuccess(const std::vector<Instruction> &cmd_list);
bool							IsVerifyOk(const NetConfigCMD &cmd);
bool							IsVerifyOk(const Instruction &cmd);
void							MakeVerify(Instruction &cmd);
std::string					    IP2Str(const IP &ip);
UINT16							Threshold2Base(UINT16 n);
UINT16							Base2Threshold(UINT16 n);
void							UINT64ToUint48(UINT64 a, UINT8 b[6]);

Instruction					    MakeResetCMD();
Instruction					    MakeReadBDMIdCMD();
Instruction                     MakeReadMAC1CMD();
Instruction                     MakeReadMAC2CMD();
Instruction                     MakeReadYearCMD();
Instruction                     MakeReadModelCMD();
Instruction					    MakeReadVersionCMD();
Instruction					    MakeReadMaxBinCMD(int nIndex);
Instruction					    MakeReadNetworkCMD(int nIndex);
Instruction					    MakeReadThresholdCMD(int nChannelId, int nThresholdId);

Instruction					    MakeWriteNetworkCMD(int nIndex, UINT16 nValue);
Instruction					    MakeWriteThresholdCMD(int nChannelId, int nThresholdId, UINT16 nValue);
Instruction                     MakePermissionCMD(bool bAdvanced);
Instruction                     MakeWriteBDMIdCMD(int nBDMId);
Instruction                     MakeWriteMAC1CMD(const UINT8 nMAC[2]);
Instruction                     MakeWriteMAC2CMD(const UINT8 nMAC[2]);
Instruction                     MakeWriteYearCMD(UINT16 nYear);
Instruction                     MakeWriteModelCMD(UINT16 nModel);
Instruction                     MakeWriteMaxBinCMD(int nMaxBinIndex, int nMaxBin);
Instruction                     MakeWriteBiasedCMD(int nChannelId, int nThreId, int nThre);
Instruction                     MakeWriteDelayCMD(int nChannelId, int nDelayId, int nDelay);

Instruction					    MakeUpdateVersionCMD();
Instruction					    MakeUpdateVersionTriggerCMD();
Instruction					    MakeUpdateVersionDataCMD(UINT8 nData1, UINT8 nData2, UINT8 nData3, UINT8 nData4);
Instruction					    MakeUpdateVersionDataOffsetCMD(int nOffset);

NetBasicCMD                     MakeBasicCMDByInstruction(Instruction cmd);
NetConfigCMD                    MakeUDPDataCMD(const std::vector<Instruction> &cmd_list, UINT64 cmd_index, BDM bdm);

void							Resolve(ResolvePulse &resolve, const Pulse &ev, UINT16 nMaxBin[3]);
void							Resolve(ResolvePackage &resolve, const Package &src, UINT16 nMaxBin[3]);


}


//std::string Command::GetCalibPrefix(int nCalibId)
//{
//    return (nCalibId == 0 ? "" : QString("%1-").arg(nCalibId).toStdString());
//}

void Command::DecMacToHex(UINT64 nDecMac, UINT8* pHexMac)
{
    memset(pHexMac, 0, sizeof(UINT8) * 6);

    for (int i = 0; i < 6; i++)
    {
        UINT64 a = pow(256, 5 - i);

        pHexMac[i] = nDecMac / a % 256;
    }
}

UINT64 Command::GetMacDec(const UINT8* pMac)
{
    UINT64 dec = 0;

    for (int i = 0; i < 6; i++)
    {
        dec += pMac[i] * pow(256, 5 - i);
    }

    return dec;
}

std::string Command::GetMacStr(const UINT8* pMac)
{
    char strMac[256] = { '\0' };

    sprintf_s(strMac, "%02X-%02X-%02X-%02X-%02X-%02X", pMac[0], pMac[1], pMac[2], pMac[3], pMac[4], pMac[5]);

    return strMac;
}


//QString Command::GetSrcFile(int nBDMId, QString qstrSavePath, QString qstrPrefix)
//{
//    QString strFile;
//
//    strFile = qstrSavePath + "/" + "BDM" + QString(nBDMId) + "_" + qstrPrefix + ".src";
//
//    return strFile;
//}
//
//QString Command::GetResolveFile(int nBDMId, QString qstrSavePath, QString qstrPrefix)
//{
//    QString strFile;
//
//    strFile = qstrSavePath + "/" + "BDM" + QString(nBDMId) + "_" + qstrPrefix + ".resolve";
//
//    return strFile;
//}
//
//QString Command::GetSrcFile(const UINT8 nMAC[6], QString qstrSavePath, QString qstrPrefix)
//{
//    QString strFile;
//    std::string mac_str = GetMacStr(nMAC);
//
//    strFile = qstrSavePath + "/" + "BDM" + QString::fromStdString(mac_str) + "_" + qstrPrefix + ".src";
//
//    return strFile;
//}
//
//QString Command::GetResolveFile(const UINT8 nMAC[6], QString qstrSavePath, QString qstrPrefix)
//{
//    QString strFile;
//    std::string mac_str = GetMacStr(nMAC);
//    strFile = qstrSavePath + "/" + "BDM" + QString::fromStdString(mac_str) + "_" + qstrPrefix + ".resolve";
//
//    return strFile;
//}

std::string Command::GetSimplified(double fNumber, int nDecimals)
{
    char strNumber[256];

    double fSimpleNumber = (fNumber > 1000000000 ? fNumber / 1000000000 : (fNumber > 1000000 ? fNumber / 1000000 : (fNumber > 1000 ? fNumber / 1000 : fNumber)));
    char c = (fNumber > 1000000000 ? 'G' : (fNumber > 1000000 ? 'M' : (fNumber > 1000 ? 'K' : ' ')));

    sprintf_s(strNumber, "%.*f%c", nDecimals, fSimpleNumber, c);

    return strNumber;
}

UINT8 Command::GetVerifyValue(UINT8 nValue1, UINT8 nValue2, UINT8 nValue3, UINT8 nValue4, UINT8 nValue5)
{
    UINT8 nSum = nValue1 + nValue2 + nValue3 + nValue4 + nValue5;

    return 0xFF - nSum;
}

ErrorType Command::ReturnType(const Instruction& cmd)
{
    switch (cmd.nOperate)
    {
    case RETURN_SUCCESS:
        return SUCCESS;
    case RETURN_ERROR:
    {
        switch (cmd.nLowData)
        {
        case RETURN_ERROR_VERIFY:
            return ERROR_VERIFY;
        case RETURN_ERROR_CMD:
            return ERROR_CMD;
        case RETURN_ERROR_TIMEOUT:
            return ERROR_TIME_OUT;
        case RETURN_ERROR_ADDRESS:
            return ERROR_ADDRESS;
        default:
            return ERROR_UNKNOWN;
        }
    }
    default:
        return ERROR_UNKNOWN;
    }
}

//bool Command::IsReturnSuccess(const std::vector<Instruction>& cmd_list)
//{
//    for (UINT64 i = 0; i < cmd_list.size(); i++)
//    {
//        if (SUCCESS != ReturnType(cmd_list[i]))
//        {
//            qDebug() << "ReturnType" << ReturnType(cmd_list[i]) << "i" << i;
//            return false;
//        }
//    }
//
//    return true;
//}

bool Command::IsVerifyOk(const Instruction& cmd)   //检查校正
{
    UINT8 nCalcVerify = GetVerifyValue(cmd.nOperate, cmd.nHighAddr, cmd.nLowAddr, cmd.nHighData, cmd.nLowData);

    return nCalcVerify == cmd.nVerify;
}

bool Command::IsVerifyOk(const NetConfigCMD& cmd)
{
    Instruction tmp;
    for (int i = 0; i < MAX_CMD_NUM; i++)
    {
        tmp = cmd.nCMDList[i].nCMD;

        if (GetVerifyValue(tmp.nOperate, tmp.nHighAddr, tmp.nLowAddr, tmp.nHighData, tmp.nLowData) != tmp.nVerify)
        {
            return false;
        }
    }

    return true;
}

void Command::MakeVerify(Instruction& cmd)
{
    cmd.nVerify = GetVerifyValue(cmd.nOperate, cmd.nHighAddr, cmd.nLowAddr, cmd.nHighData, cmd.nLowData);
}

std::string Command::IP2Str(const IP& ip)
{
    char str[256];

    sprintf_s(str, "%d.%d.%d.%d", ip.a, ip.b, ip.c, ip.d);

    return str;
}

UINT16 Command::Threshold2Base(UINT16 n)
{
    UINT16 base;
    base = static_cast<UINT16>(double(n) * 4096 / 1250 + 0.5);
    return base;
}

UINT16 Command::Base2Threshold(UINT16 n)
{
    UINT16 threshold;
    threshold = static_cast<UINT16>(double(n) * 1250 / 4096 + 0.5);
    return threshold;
}


Instruction Command::MakeResetCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_RESET;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadBDMIdCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_BDMID;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadMAC1CMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_BDMID;
    cmd.nLowAddr = 0x00;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadMAC2CMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_BDMID;
    cmd.nLowAddr = 0x01;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadYearCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_BDMID;
    cmd.nLowAddr = 0x04;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadModelCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_BDMID;
    cmd.nLowAddr = 0x03;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteMAC1CMD(const UINT8 nMAC[2])
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BDMID;
    cmd.nLowAddr = 0x00;
    cmd.nHighData = nMAC[0];
    cmd.nLowData = nMAC[1];

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteMAC2CMD(const UINT8 nMAC[2])
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BDMID;
    cmd.nLowAddr = 0x01;
    cmd.nHighData = nMAC[0];
    cmd.nLowData = nMAC[1];

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteYearCMD(UINT16 nYear)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BDMID;
    cmd.nLowAddr = 0x04;
    cmd.nHighData = nYear / 256;
    cmd.nLowData = static_cast<UINT8>(nYear % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteModelCMD(UINT16 nModel)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BDMID;
    cmd.nLowAddr = 0x03;
    cmd.nHighData = nModel / 256;
    cmd.nLowData = static_cast<UINT8>(nModel % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadVersionCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_VERSION;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadMaxBinCMD(int nIndex)
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_MAXBIN;
    cmd.nLowAddr = static_cast<UINT8>(nIndex);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadNetworkCMD(int nIndex)
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_NETWORK;
    cmd.nLowAddr = static_cast<UINT8>(nIndex);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeReadThresholdCMD(int nChannelId, int nThresholdId)
{
    Instruction cmd;
    cmd.nOperate = SEND_READ_CONFTHRESHOLD;
    cmd.nHighAddr = static_cast<UINT8>(nChannelId);
    cmd.nLowAddr = static_cast<UINT8>(nThresholdId);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteNetworkCMD(int nIndex, UINT16 nValue)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_NETWORK;
    cmd.nLowAddr = static_cast<UINT8>(nIndex);
    cmd.nHighData = nValue / 256;
    cmd.nLowData = static_cast<UINT8>(nValue % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteThresholdCMD(int nChannelId, int nThresholdId, UINT16 nValue)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_CONFTHRESHOLD;
    cmd.nHighAddr = static_cast<UINT8>(nChannelId);
    cmd.nLowAddr = static_cast<UINT8>(nThresholdId);
    cmd.nHighData = nValue / 256;
    cmd.nLowData = static_cast<UINT8>(nValue % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakePermissionCMD(bool bAdvanced)
{
    Instruction cmd;
    cmd.nOperate = SEND_PERMISSTION;
    cmd.nLowData = (bAdvanced ? PERMISSTION_OPEN : PERMISSTION_CLOSE);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteBDMIdCMD(int nBDMId)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BDMID;
    cmd.nHighData = static_cast<UINT8>(nBDMId / 256);
    cmd.nLowData = static_cast<UINT8>(nBDMId % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteMaxBinCMD(int nMaxBinIndex, int nMaxBin)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_MAXBIN;
    cmd.nLowAddr = static_cast<UINT8>(nMaxBinIndex);
    cmd.nLowData = static_cast<UINT8>(nMaxBin % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeWriteBiasedCMD(int nChannelId, int nThreId, int nThre)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_BIASEDTHRESHOLD;
    cmd.nHighAddr = static_cast<UINT8>(nChannelId);
    cmd.nLowAddr = static_cast<UINT8>(nThreId);
    cmd.nHighData = static_cast<UINT8>(nThre / 256);
    cmd.nLowData = static_cast<UINT8>(nThre % 256);

    MakeVerify(cmd);
    return cmd;
}

Instruction Command::MakeWriteDelayCMD(int nChannelId, int nDelayId, int nDelay)
{
    Instruction cmd;
    cmd.nOperate = SEND_WRITE_DELAY;
    cmd.nHighAddr = static_cast<UINT8>(nChannelId);
    cmd.nLowAddr = static_cast<UINT8>(nDelayId);
    cmd.nHighData = static_cast<UINT8>(nDelay / 256);
    cmd.nLowData = static_cast<UINT8>(nDelay % 256);

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeUpdateVersionCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_UPDATE_FIRMWARE;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeUpdateVersionTriggerCMD()
{
    Instruction cmd;
    cmd.nOperate = SEND_UPDATE_FIRMWARE_TRIGGER;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeUpdateVersionDataCMD(UINT8 nData1, UINT8 nData2, UINT8 nData3, UINT8 nData4)
{
    Instruction cmd;
    cmd.nOperate = SEND_UPDATE_FIRMWARE_DATA;
    cmd.nHighAddr = nData4;
    cmd.nLowAddr = nData3;
    cmd.nHighData = nData2;
    cmd.nLowData = nData1;

    MakeVerify(cmd);

    return cmd;
}

Instruction Command::MakeUpdateVersionDataOffsetCMD(int nOffset)
{
    Instruction cmd;
    cmd.nOperate = SEND_UPDATE_FIRMWARE_DATA_OFFSET;
    cmd.nHighAddr = UINT8(nOffset >> 24);
    cmd.nLowAddr = UINT8(nOffset >> 16);
    cmd.nHighData = UINT8(nOffset >> 8);
    cmd.nLowData = UINT8(nOffset);

    MakeVerify(cmd);

    return cmd;
}

NetBasicCMD Command::MakeBasicCMDByInstruction(Instruction cmd)
{
    NetBasicCMD basic_cmd;
    basic_cmd.nHead[0] = 0x00;
    basic_cmd.nHead[1] = 0x00;
    basic_cmd.nCMD = cmd;
    return basic_cmd;
}

NetConfigCMD Command::MakeUDPDataCMD(const std::vector<Instruction>& cmd_list, UINT64 cmd_index, BDM bdm)
{
    NetConfigCMD cmd;
    for (int i = 0; i < 4; i++) {
        cmd.nMAC[i] = bdm.nMAC[i + 2];
    }
    for (int i = 0; i < MAX_CMD_NUM; i++)
    {
        cmd.nCMDList[i].nCMD.nOperate = 0xFF;
    }

    UINT64 cmd_count = 0;

    for (UINT64 i = 0; i < MAX_CMD_NUM; i++)
    {
        cmd_count = i + cmd_index * MAX_CMD_NUM;
        if (cmd_count >= cmd_list.size())
        {
            break;
        }
        cmd.nCMDList[i] = MakeBasicCMDByInstruction(cmd_list[static_cast<UINT64>(i + cmd_index * MAX_CMD_NUM)]);
    }
    cmd.nTailFlag[0] = 0x01;
    cmd.nTailFlag[1] = 0x2C;
    cmd.nTailFlag[2] = 0x00;
    cmd.nTailFlag[3] = 0x00;

    return  cmd;

}

void Command::SwitchLeBe(UINT32& a)
{
    a = (a << 24 & 0xFF000000) | (a << 8 & 0x00FF0000) | (a >> 8 & 0x0000FF00) | (a >> 24 & 0x000000FF);
}

void Command::SwitchLeBe(UINT16& a)
{
    a = (a << 8 & 0xFF00) | (a >> 8 & 0x00FF);
}

void Command::SwitchLeBe(UINT64& a)
{
    a = (a << 56 & 0xFF00000000000000) | (a << 40 & 0x00FF000000000000) | (a << 24 & 0x0000FF0000000000) | (a << 8 & 0x000000FF00000000) |
        (a >> 8 & 0x00000000FF000000) | (a >> 24 & 0x0000000000FF0000) | (a >> 40 & 0x000000000000FF00) | (a >> 56 & 0x00000000000000FF);
}


void Command::SwitchLeBe(NetQueryReceiveCMD& cmd)
{
    SwitchLeBe(cmd.nPort);
}


void Command::UINT64ToUint48(UINT64 a, UINT8 b[6])
{
    b[0] = UINT8(a);
    b[1] = UINT8(a >> 8);
    b[2] = UINT8(a >> 16);
    b[3] = UINT8(a >> 24);
    b[4] = UINT8(a >> 32);
    b[5] = UINT8(a >> 40);
}

void Command::Resolve(ResolvePackage& resolve, const Package& src, UINT16 nMaxBin[3])
{
    for (int i = 0; i < EVENT_NUM_PACKAGE; i++)
    {
        //std::cout << "原始时间: ";
        Resolve(resolve.nEvents[i], src.nEvents[i], nMaxBin);
        // std::cout << std::endl;
    }
}

void Command::Resolve(ResolvePulse& resolve, const Pulse& ev, UINT16 nMaxBin[3])
{
    assert(ev.nData[3] >= 1 && ev.nData[3] <= 72);

    resolve.nChannelId = ev.nData[3];

    int nCurMaxBin = (ev.nData[3] <= 24 ? nMaxBin[0] : (ev.nData[3] <= 48 ? nMaxBin[1] : nMaxBin[2]));
    double fTmp = ev.nData[7] * 4294967296.0 + ev.nData[6] * 16777216.0 + ev.nData[5] * 65536.0 + ev.nData[4] * 256.0;
    //计算事件的时间点
    resolve.nTime[0] = (fTmp + ev.nData[9]) * 5.0 - (ev.nData[8] - 128) * 5.0 / nCurMaxBin;
    resolve.nTime[1] = (fTmp + ev.nData[13]) * 5.0 - (ev.nData[12] - 128) * 5.0 / nCurMaxBin;
    resolve.nTime[2] = (fTmp + ev.nData[17]) * 5.0 - (ev.nData[16] - 128) * 5.0 / nCurMaxBin;
    resolve.nTime[3] = (fTmp + ev.nData[21]) * 5.0 - (ev.nData[20] - 128) * 5.0 / nCurMaxBin;
    resolve.nTime[4] = (fTmp + ev.nData[23]) * 5.0 - (192 - ev.nData[22]) * 5.0 / nCurMaxBin;
    resolve.nTime[5] = (fTmp + ev.nData[19]) * 5.0 - (192 - ev.nData[18]) * 5.0 / nCurMaxBin;
    resolve.nTime[6] = (fTmp + ev.nData[15]) * 5.0 - (192 - ev.nData[14]) * 5.0 / nCurMaxBin;
    resolve.nTime[7] = (fTmp + ev.nData[11]) * 5.0 - (192 - ev.nData[10]) * 5.0 / nCurMaxBin;

    resolve.nTime[1] += (ev.nData[13] - ev.nData[9] < -50 ? 1280 : (ev.nData[13] - ev.nData[9] > 50 ? -1280 : 0));
    resolve.nTime[2] += (ev.nData[17] - ev.nData[9] < -50 ? 1280 : (ev.nData[17] - ev.nData[9] > 50 ? -1280 : 0));
    resolve.nTime[3] += (ev.nData[21] - ev.nData[9] < -50 ? 1280 : (ev.nData[21] - ev.nData[9] > 50 ? -1280 : 0));
    resolve.nTime[4] += (ev.nData[23] < ev.nData[9] ? 1280 : 0);
    resolve.nTime[5] += (ev.nData[19] < ev.nData[9] ? 1280 : 0);
    resolve.nTime[6] += (ev.nData[15] < ev.nData[9] ? 1280 : 0);
    resolve.nTime[7] += (ev.nData[11] < ev.nData[9] ? 1280 : 0);
    //std::cout << std::endl;
    //for (int i = 0; i < 8; ++i) {
    //    std::cout << std::fixed << resolve.nTime[i] << "  " ;
    //}
    //std::cout << std::endl;
}

#endif // COMMAND_H
