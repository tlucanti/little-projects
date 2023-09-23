#include "out.h"



int _init(EVP_PKEY_CTX *ctx)

{
  int iVar1;

  iVar1 = __gmon_start__();
  return iVar1;
}



void FUN_00401020(void)

{
                    // WARNING: Treating indirect jump as call
  (*(code *)(undefined *)0x0)();
  return;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

DIR * opendir(char *__name)

{
  DIR *pDVar1;

  pDVar1 = opendir(__name);
  return pDVar1;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

int printf(char *__format,...)

{
  int iVar1;

  iVar1 = printf(__format);
  return iVar1;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

int closedir(DIR *__dirp)

{
  int iVar1;

  iVar1 = closedir(__dirp);
  return iVar1;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

int strcmp(char *__s1,char *__s2)

{
  int iVar1;

  iVar1 = strcmp(__s1,__s2);
  return iVar1;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

dirent * readdir(DIR *__dirp)

{
  dirent *pdVar1;

  pdVar1 = readdir(__dirp);
  return pdVar1;
}



// WARNING: Unknown calling convention -- yet parameter storage is locked

int usleep(__useconds_t __useconds)

{
  int iVar1;

  iVar1 = usleep(__useconds);
  return iVar1;
}



void processEntry _start(undefined8 param_1,undefined8 param_2)

{
  undefined auStack_8 [8];

  __libc_start_main(main,param_2,&stack0x00000008,0,0,param_1,auStack_8);
  do {
                    // WARNING: Do nothing block with infinite loop
  } while( true );
}



void _dl_relocate_static_pie(void)

{
  return;
}



// WARNING: Removing unreachable block (ram,0x004010e3)
// WARNING: Removing unreachable block (ram,0x004010ef)

void deregister_tm_clones(void)

{
  return;
}



// WARNING: Removing unreachable block (ram,0x00401124)
// WARNING: Removing unreachable block (ram,0x00401130)

void register_tm_clones(void)

{
  return;
}



void __do_global_dtors_aux(void)

{
  if (completed_0 == '\0') {
    deregister_tm_clones();
    completed_0 = 1;
    return;
  }
  return;
}



// WARNING: Removing unreachable block (ram,0x00401124)
// WARNING: Removing unreachable block (ram,0x00401130)

void frame_dummy(void)

{
  return;
}



void del(char *dir,char *name)

{
  char *name_local;
  char *dir_local;

  printf("%s/%s\n",dir,name);
  return;
}



void checkDir(char *scan_dir,int delay,char *search_name)

{
  int iVar1;
  DIR *__dirp;
  dirent *pdVar2;
  char *search_name_local;
  int delay_local;
  char *scan_dir_local;
  dirent *dir;
  DIR *d;

  while (__dirp = opendir(scan_dir), __dirp != (DIR *)0x0) {
    while (pdVar2 = readdir(__dirp), pdVar2 != (dirent *)0x0) {
      usleep(delay);
      iVar1 = strcmp(pdVar2->d_name,search_name);
      if (iVar1 == 0) {
        del(scan_dir,search_name);
      }
    }
    closedir(__dirp);
  }
  return;
}



int main(int argc,char **argv)

{
  char **argv_local;
  int argc_local;

  checkDir(".",0x5dd,"test1");
  return 0;
}



void _fini(void)

{
  return;
}




