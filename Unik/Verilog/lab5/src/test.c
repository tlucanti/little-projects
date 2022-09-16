
#include <verilog.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <color.h>

# undef KUZN_ADDRESS
# undef KUZN_ACTIVE

int all = 0, ok = 0;
# undef assert
# define assert(__expr, __msg) do { \
    ++all; \
    if (!(__expr)) { \
        printf(ERROR "[FAIL] in test " __msg RESET "\n"); \
        printf(WARNING "bus value " RESET "'%.36s'\n", bus); \
    } else { ++ok; } \
} while (0)

# define result() do { \
    if (all == ok) { \
        printf(OK "[ OK ] ALL " TERM_CYAN "%d" OK " TESTS PASSED" RESET "\n", all); \
    } else { \
        printf(ERROR "[FAIL] " TERM_CYAN" %d / %d " ERROR "TESTS PASSED" RESET "\n", ok, all); \
    } \
} while (0)

int main()
{
    unsigned char KUZN_ACTIVE;
    unsigned char *bus = (unsigned char *)malloc(0x24);
    void *KUZN_ADDRESS = bus;
    memset(bus, '0', 0x24);

    KUZN_ACTIVE = 'w';
    reset_kuzn();
    assert(bus[0] == KUZN_ACTIVE, "reset test 1");
    assert(!memcmp(bus, "w00000000000000000000000000000000000", 0x24), "reset test 2");

    KUZN_ACTIVE = 'z';
    req_kuzn();
    assert(bus[1] == KUZN_ACTIVE, "req test 1");
    assert(!memcmp(bus, "wz0000000000000000000000000000000000", 0x24), "req test 2");

    bus[2] = 'v';
    assert(valid_kuzn() == 'v', "valid test 1");
    assert(!memcmp(bus, "wzv000000000000000000000000000000000", 0x24), "valid test 2");

    bus[3] = 'b';
    assert(busy_kuzn() == 'b', "busy test 1");
    assert(!memcmp(bus, "wzvb00000000000000000000000000000000", 0x24), "busy test 2");

    encode_kuzn("O123456789ABCDEF");
    assert(!memcmp(bus, "wzvbO123456789ABCDEF0000000000000000", 0x24), "encode test 1");

    unsigned char dc[0x24];
    memset(dc, '0', 0x24);
    memcpy(bus + 0x14, "O123456789ABCDEF", 0x10);
    decode_kuzn(dc);
    assert(!memcmp(bus, "wzvbO123456789ABCDEFO123456789ABCDEF", 0x24), "decode test 1");
    assert(!memcmp(dc,  "O123456789ABCDEF000000000000000000000", 0x24), "decode test 2");

    result();
    free(bus);
}
