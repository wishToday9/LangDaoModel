#ifndef PARA_H
#define PARA_H

#include <string>
#include <string.h>

#define RELEASE_POINTER(p)  if(p) {delete p; p = nullptr;}
#define RELEASE_ARRAY_POINTER(p) if(p) {delete [] p;p = nullptr;}

struct EnergyMessage
{
    double e_time;
    double e_energy;
};
struct EnergyCount//单个柱形数据
{
    unsigned long count = 0;
    double min_energy = 0;
    double max_energy = 0;
};

struct EnergySpectrum   ///能谱
{
    double diff_count = 0;
    double diff_energy = 20;
    double minimum_energy,maxmum_energy = 0;
    unsigned int bin_num =0;//bin个数
    //int count_pixel = 2000;//纵坐标高度
    EnergyCount *bin  = nullptr;
};

struct  EnegryCorrectData
{
    uint16_t IP_CH[2];
    float par[7];
};

struct  TimeCorrectData
{
    uint16_t IP_CH[2];
    float offset;

};
typedef unsigned char		UINT8;
typedef char                INT8;
typedef unsigned long long	UINT64;
typedef long long           INT64;
typedef	unsigned int		UINT32;
typedef unsigned short		UINT16;
typedef short               INT16;

const double    PI          = 3.14;
const double    EPSILON     = 0.0000001;

const int MAX_COM_NUM		= 256;
const int BAUDRATE_NUM      = 10;
const int BYTESIZE_NUM      = 3;
const int PARITY_NUM        = 5;
const int STOPBITS_NUM      = 3;

const int CHANNEL_NUM       = 72;//通道数1个
const int FitEvent_NUM      = 6000;
const int THRESHOLD_NUM     = 4;
const int BIASED_NUM        = 4;//偏置的数量？
const int DELAY_NUM         = 8;
const int MAXBIN_NUM        = 3;
const int MAX_CMD_NUM       = 144;//每个UDP包有144条指令（8字节）
const int MAX_BUFFER_SIZE   = 1000 * 1024 * 1024;//内存大小
const int EVENT_NUM_PACKAGE = 48;

const std::string LOG_PATH = "./Log/";

const UINT8 SEND_RESET							= 0x00;
const UINT8 SEND_READ_VERSION					= 0x02;
const UINT8 SEND_PERMISSTION					= 0x05;
const UINT8 SEND_UPDATE_FIRMWARE				= 0x08;
const UINT8 SEND_WRITE_CONFTHRESHOLD			= 0x10;
const UINT8 SEND_READ_CONFTHRESHOLD				= 0x20;
const UINT8 SEND_WRITE_NETWORK					= 0x11;
const UINT8 SEND_READ_NETWORK					= 0x21;
const UINT8 SEND_WRITE_BIASEDTHRESHOLD			= 0x14;
const UINT8 SEND_READ_BIASEDTHRESHOLD			= 0x24;
const UINT8 SEND_WRITE_DELAY					= 0x15;
const UINT8 SEND_READ_DELAY						= 0x25;
const UINT8 SEND_WRITE_BDMID					= 0x16;
const UINT8 SEND_READ_BDMID						= 0x26;
const UINT8 SEND_WRITE_MAXBIN					= 0x17;
const UINT8 SEND_READ_MAXBIN					= 0x27;
const UINT8 SEND_UPDATE_FIRMWARE_TRIGGER		= 0x30;
const UINT8 SEND_UPDATE_FIRMWARE_DATA			= 0x31;
const UINT8 SEND_UPDATE_FIRMWARE_DATA_OFFSET	= 0x32;

const UINT8 RETURN_SUCCESS						= 0x01;
const UINT8 RETURN_ERROR						= 0X10;
const UINT8 RETURN_ERROR_VERIFY					= 0x02;
const UINT8 RETURN_ERROR_CMD					= 0x05;
const UINT8 RETURN_ERROR_ADDRESS                = 0x06;
const UINT8 RETURN_ERROR_TIMEOUT				= 0x08;

const UINT8 PERMISSTION_OPEN					= 0x00;
const UINT8 PERMISSTION_CLOSE					= 0x08;

enum SoftwareStage
{
    STAGE_BASE = 0,
    STAGE_ALPHA,
    STAGE_BETA,
    STAGE_RC,
    STAGE_RELEASE
};
const std::string StageName[] =
{
    "base",
    "alpha",
    "beta",
    "rc",
    "release"
};
struct SoftwareVersion
{
    int             nMajor;
    int             nMinor;
    int             nRevision;
    SoftwareStage   nStage;
};

enum Permission
{
    PERMISSION_NORMAL = 0,
    PERMISSION_ADVANCED
};

typedef struct _IP
{
    UINT8			a;
    UINT8			b;
    UINT8			c;
    UINT8			d;
}IP;

typedef _IP Version;

struct Network
{
    IP				nSrcIP;
    UINT16			nSrcPort;
    IP				nDstIP;
    UINT16			nDstPort;
};

typedef UINT8 MACAddr[6];

struct Socket
{
    IP				nIP;
    UINT32			nPort;
};

struct Instruction
{
    UINT8			nOperate;
    UINT8			nHighAddr;
    UINT8			nLowAddr;
    UINT8			nHighData;
    UINT8			nLowData;
    UINT8			nVerify;

    Instruction()
    {
        nOperate	= 0x00;
        nHighAddr	= 0x00;
        nLowAddr	= 0x00;
        nHighData	= 0x00;
        nLowData	= 0x00;
        nVerify		= 0x00;
    }
};

struct Pulse
{
    UINT8			nData[24];//UDP包中一个事件24个字节
};

struct Package
{
    UINT8			nHead[2];
    UINT8			nBDMId[2];
    Pulse			nEvents[EVENT_NUM_PACKAGE];
    UINT8			nTail[2];
    //uint16_t          nTail[1];
};

#pragma pack(push, 1)
struct ResolvePulse
{
    UINT8			nChannelId;
    double			nTime[8];//一个事件16个采样点（8个阈值）
};
struct ResolvePackage
{
    ResolvePulse    nEvents[EVENT_NUM_PACKAGE];
};
#pragma pack(pop)

struct BDMCount
{
    int             nBDMId;
    double          fCounts;
};

struct BDM
{
    UINT16          nBDMId;
    UINT8			nMAC[6];
    UINT16          nYear;
    UINT16          nModel;
    Network			nNetwork;
    Version			nVersion;
    UINT16			nMaxBin[3];
    UINT16			nThreshold[CHANNEL_NUM][THRESHOLD_NUM];

};

struct SimpleBDM
{
    UINT16			nBDMId;
    UINT8           nMAC[6];
    Network			nNetwork;
    UINT16			nMaxBin[3];
};

struct NetQuerySendCMD
{
    UINT8			nMAC[4];
    UINT8			nAction[8];
    UINT8			nTail[1148];
};

struct NetQueryReceiveCMD
{
    UINT8			nMAC[4];
    UINT8			nSrcMAC[6];
    IP				nIP;
    UINT16			nPort;
    UINT8			nTail[1140];
    UINT8			nTailFlag[4];
};

struct NetBasicCMD                             //each command is 6 Byte and add 2 Byte head 0000H
{
    UINT8			nHead[2];
    Instruction		nCMD;
};

struct NetConfigCMD                             // 4 Byte Mac, 144 command, 4 tail flag 012C0000H
{
    UINT8			nMAC[4];
    NetBasicCMD	nCMDList[MAX_CMD_NUM];
    UINT8			nTailFlag[4];
};

//struct SinglesStruct
//{
//    double energy[EVENT_NUM_PACKAGE];
//};
typedef struct _SinglesStruct
{
    UINT32    globalCrystalIndex;
    float     energy;
    double    timevalue;
}SinglesStruct;

/* Used in listmode */
typedef struct _CoinStruct
{
    SinglesStruct nCoinStruct[2];
}CoinStruct;


struct EnergyStruct
{
    double energy;
};
enum  ErrorType
{
    SUCCESS		= 0,
    ERROR_STOPPED,
    ERROR_FILE_OPEN,

    ERROR_SERIAL_OPEN,
    ERROR_SERIAL_WRITE,

    ERROR_NET_WRITE,
    ERROR_NET_BUSYORWRONG,

    ERROR_GATHER_NO_DATA,

    ERROR_VERIFY,
    ERROR_CMD,
    ERROR_ADDRESS,
    ERROR_TIME_OUT,

    ERROR_UNKNOWN
};

static const char *ErrorMsg[] =
{
    "Success! Please waiting for data processing...",
    "The progress is stopped!",
    "Failed to open file!",

    "Failed to open serial!",
    "Failed to write by serial!",

    "Failed to write by network!",
    "The network is busy or wrong!",

    "There is no data transfering, please check the connection!",

    "Error in verifying cmd!",
    "Error cmd!",
    "Cmd Timeout!",

    "unknown error!",
};


enum  GatherType
{
    GATHER_TIME = 0,
    GATHER_COUNT,
    GATHER_MANUALLY
};

enum OperateType
{
    OPERATE_THRESHOLD_USING_SERIAL = 0,
    OPERATE_BIAS_USING_SERIAL,
    OPERATE_DELAY_USING_SERIAL,
    OPERATE_FIRMWARE_USING_SERIAL,
    OPERATE_INIT_CALIBRATION_SERIAL,
    OPERATE_THRESHOLD_USING_NET_ONEBDM,
    OPERATE_BIAS_USING_NET_ONEBDM,
    OPERATE_DELAY_USING_NET_ONEBDM,
    OPERATE_FIRMWARE_USING_NET_ONEBDM,
    OPERATE_INIT_CALIBRATION_NET_ONEBDM,
    OPERATE_THRESHOLD_USING_NET_ALLBDM,
    OPERATE_FIRMWARE_USING_NET_ALLBDM,
    OPERATE_AUTO_CALIBRATION,
    OPERATE_GATHER
};

enum ConfigType
{
    SERIAL_CONFIG = 0,
    NET_CONFIG
};

enum ProcessType
{
    PROCESS_NORMAL = 0,
    PROCESS_HIGH_SPEED,
    PROCESS_NO_RESOLVE
};

enum NormalizationType
{
    NONE = 0,
    SOURCE_Cs
};


const IP        LOCAL_IP = {192, 168, 1, 110};
const UINT16    LOCAL_CONFIG_PORT = 15000;
const IP        BROADCAST_IP = { 255, 255, 255, 255 };
const UINT16    BROADCAST_PORT = 65535;

static const char *CALIBRATION_STATUS[] = {"None", "Delay", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8"};

#endif // PARA_H

