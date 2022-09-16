
#ifndef MAIN_H
# define MAIN_H

# include <stdint.h>

/*
     address  type  info
    0x1000000   W   reset address
    0x1000001   W   req, ack signal
    0x1000002   R   valid
    0x1000003   R   busy
    0x1000004   W   write address 16 bytes
    0x1000005   |
    0x1000006   |
    0x1000007   |
    0x1000008   |
    0x1000009   |
    0x100000A   |
    0x100000B   |
    0x100000C   |
    0x100000D   |
    0x100000E   |
    0x100000F   |
    0x1000010   |
    0x1000011   |
    0x1000012   |
    0x1000013   |
    0x1000014   R   read address 16 bytes
    0x1000015   |
    0x1000016   |
    0x1000017   |
    0x1000018   |
    0x1000019   |
    0x100001A   |
    0x100001B   |
    0x100001C   |
    0x100001D   |
    0x100001E   |
    0x100001F   |
    0x1000020   |
    0x1000021   |
    0x1000022   |
    0x1000023   |
*/
# define KUZN_ACTIVE            0x0

# define KUZN_ADDRESS           (uint8_t *)(0x1000000)
# define KUZN_RESET_ADDRESS     (uint8_t *)(KUZN_ADDRESS + 0x00)
# define KUZN_REQ_ADDRESS       (uint8_t *)(KUZN_ADDRESS + 0x01)
# define KUZN_VALID_ADDRESS     (uint8_t *)(KUZN_ADDRESS + 0x02)
# define KUZN_BUSY_ADDRESS      (uint8_t *)(KUZN_ADDRESS + 0x03)
# define KUZN_INPUT_ADDRESS     (uint8_t *)(KUZN_ADDRESS + 0x04)
# define KUZN_OUTPUT_ADDRESS    (uint8_t *)(KUZN_ADDRESS + 0x14)

# define reset_kuzn()           do { *KUZN_RESET_ADDRESS = KUZN_ACTIVE; } while (0)
# define req_kuzn()             do { *KUZN_REQ_ADDRESS = KUZN_ACTIVE; } while (0)
# define valid_kuzn()           (*KUZN_VALID_ADDRESS)
# define busy_kuzn()            (*KUZN_BUSY_ADDRESS)
# define encode_kuzn(__PTR)     do { \
                                    ((uint32_t *)KUZN_INPUT_ADDRESS)[0] = ((uint32_t *)__PTR)[0]; \
                                    ((uint32_t *)KUZN_INPUT_ADDRESS)[1] = ((uint32_t *)__PTR)[1]; \
                                    ((uint32_t *)KUZN_INPUT_ADDRESS)[2] = ((uint32_t *)__PTR)[2]; \
                                    ((uint32_t *)KUZN_INPUT_ADDRESS)[3] = ((uint32_t *)__PTR)[3]; \
                                } while (0)
# define decode_kuzn(__PTR)     do { \
                                    ((uint32_t *)__PTR)[0] = ((uint32_t *)(KUZN_OUTPUT_ADDRESS))[0]; \
                                    ((uint32_t *)__PTR)[1] = ((uint32_t *)(KUZN_OUTPUT_ADDRESS))[1]; \
                                    ((uint32_t *)__PTR)[2] = ((uint32_t *)(KUZN_OUTPUT_ADDRESS))[2]; \
                                    ((uint32_t *)__PTR)[3] = ((uint32_t *)(KUZN_OUTPUT_ADDRESS))[3]; \
                                } while (0)

#endif /* MAIN_H */
