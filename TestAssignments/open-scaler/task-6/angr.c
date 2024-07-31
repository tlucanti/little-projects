
long long _init()
{
    if (true)
        return 0;
    return 0();
}

long long sub_401020()
{
    void* v0;  // [bp-0x8]

    v0 = 0;
    goto *((long long *)4210704);
}

extern unsigned long long main;

long long _start()
{
    char v0;  // [bp+0x0], Other Possible Types: unsigned long
    unsigned long v1;  // [bp+0x8]
    unsigned long long v2;  // rsi
    unsigned long v3;  // rax
    unsigned long long v4;  // rdx

    v2 = *((long long *)&v0);
    v0 = v3;
    __libc_start_main(&main, v2, &v1, 0x0, 0x0, v4); /* do not return */
}

// No decompilation output for function sub_4010b5

long long _dl_relocate_static_pie()
{
    unsigned long v1;  // rax

    return v1;
}

extern char __TMC_END__;

int deregister_tm_clones()
{
    if (true)
        return;
    if (!(false))
        return;
}

long long register_tm_clones()
{
    if (true)
        return 0;
    if (!(false))
        return 0;
}

extern char __TMC_END__;

long long __do_global_dtors_aux()
{
    unsigned long v0;  // [bp-0x8]
    char v1;  // [bp+0x0]
    unsigned long v3;  // rax

    if (__TMC_END__)
        return v3;
    v0 = &v1;
    __TMC_END__ = 1;
    return deregister_tm_clones();
}

long long frame_dummy()
{
    return register_tm_clones();
}

int del(unsigned long a0, unsigned long a1)
{
    printf("%s/%s\n", (unsigned int)a0, (unsigned int)a1);
    return;
}

int checkDir(unsigned long a0, unsigned int a1, unsigned long a2)
{
    unsigned long v0;  // [bp-0x18]
    char *v1;  // [bp-0x10]

    while (true)
    {
        v1 = opendir(a0);
        if (!v1)
            break;
        while (true)
        {
            v0 = readdir(v1);
            if (!v0)
                break;
            usleep(a1);
            if (!strcmp(v0 + 19, a2))
                del(a0, a2);
        }
        closedir(v1);
    }
    return;
}

int main(unsigned int a0, unsigned long a1)
{
    unsigned long v0;  // [bp-0x18]
    unsigned int v1;  // [bp-0xc]

    v1 = a0;
    v0 = a1;
    checkDir(".", 0x5dd, "test1");
    return 0;
}

long long _fini()
{
    unsigned long v1;  // rax

    return v1;
}


