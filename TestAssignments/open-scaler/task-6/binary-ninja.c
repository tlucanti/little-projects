void _init()
{
    if (__gmon_start__ != 0)
    {
        __gmon_start__();
    }
}

int64_t sub_401020()
{
    int64_t var_8 = data_404008;
    /* jump -> data_404010 */
}

DIR* opendir(char const* name)
{
    /* tailcall */
    return opendir(name);
}

int64_t sub_401036()
{
    int64_t var_8 = 0;
    /* tailcall */
    return sub_401020();
}

int32_t printf(char const* format, ...)
{
    /* tailcall */
    return printf();
}

int64_t sub_401046()
{
    int64_t var_8 = 1;
    /* tailcall */
    return sub_401020();
}

int32_t closedir(DIR* dirp)
{
    /* tailcall */
    return closedir(dirp);
}

int64_t sub_401056()
{
    int64_t var_8 = 2;
    /* tailcall */
    return sub_401020();
}

int32_t strcmp(char const* arg1, char const* arg2)
{
    /* tailcall */
    return strcmp(arg1, arg2);
}

int64_t sub_401066()
{
    int64_t var_8 = 3;
    /* tailcall */
    return sub_401020();
}

struct dirent64* readdir(DIR* dirp)
{
    /* tailcall */
    return readdir(dirp);
}

int64_t sub_401076()
{
    int64_t var_8 = 4;
    /* tailcall */
    return sub_401020();
}

int32_t usleep(useconds_t useconds)
{
    /* tailcall */
    return usleep(useconds);
}

int64_t sub_401086()
{
    int64_t var_8 = 5;
    /* tailcall */
    return sub_401020();
}

int64_t _start(int64_t arg1, int64_t arg2, void (* arg3)()) __noreturn
{
    int64_t stack_end_1;
    int64_t stack_end = stack_end_1;
    __libc_start_main(main, __return_addr, &ubp_av, nullptr, nullptr, arg3, &stack_end);
    /* no return */
}

int64_t _dl_relocate_static_pie() __pure
{
    return;
}

void deregister_tm_clones()
{
    return;
}

void register_tm_clones()
{
    return;
}

void __do_global_dtors_aux()
{
    if (__TMC_END__ != 0)
    {
        return;
    }
    deregister_tm_clones();
    __TMC_END__ = 1;
}

void frame_dummy()
{
    /* tailcall */
    return register_tm_clones();
}

int64_t del(int64_t arg1, int64_t arg2)
{
    return printf("%s/%s\n", arg1, arg2);
}

DIR* checkDir(char* arg1, useconds_t arg2, char* arg3)
{
    DIR* dirp;
    while (true)
    {
        dirp = opendir(arg1);
        if (dirp == 0)
        {
            break;
        }
        while (true)
        {
            struct dirent64* rax_8 = readdir(dirp);
            if (rax_8 == 0)
            {
                break;
            }
            usleep(arg2);
            if (strcmp(&rax_8->d_name, arg3) == 0)
            {
                del(arg1, arg3);
            }
        }
        closedir(dirp);
    }
    return dirp;
}

int32_t main(int32_t argc, char** argv, char** envp)
{
    int32_t argc_1 = argc;
    char** argv_1 = argv;
    checkDir(&data_402011, 0x5dd, "test1");
    return 0;
}

int64_t _fini() __pure
{
    return;
}


