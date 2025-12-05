#pragma once

// Forward declarations - will be defined by panda
#define CANPACKET_HEAD_SIZE 6U
#define CANPACKET_DATA_SIZE_MAX 64U
#define CAN_PACKET_VERSION 4

// Use forward declaration to avoid conflicts with panda/board/can_declarations.h
typedef struct CANPacket_s CANPacket_t;

#define GET_LEN(msg) (dlc_to_len[(msg)->data_len_code])
