#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np

import subprocess
import os
from PIL import Image

FLAGS = ['-O2']

def run(elf, flags=None):
    if flags is None:
        flags = []

    try:
        print(subprocess.check_output(['./' + elf] + flags, stderr=subprocess.STDOUT).decode('utf-8'), end='')
    except subprocess.CalledProcessError as e:
        print('output:')
        print(e.output.decode('utf-8'), end='')
        raise


def compile(name, flags=[]):
    cu = name + '.cu'
    out = name + '.elf'

    print(f'compilation log of {cu} ', end='')
    print(subprocess.check_output(['nvcc', '-Wno-deprecated-declarations', '-I .', cu, '-o', out] + FLAGS + flags).decode('utf-8'), end='')
    print('OK')


def run_lab(lab, task, flags=[]):
    compile(f'lab{lab}/task{task}', flags)
    run(f'lab{lab}/task{task}.elf')


def convert(src, dst):
    with Image.open(src) as im:
            im.save(dst)


class Lab:
    class Lab1:
        def task1():
            run_lab(lab=1, task=1)

        def task2():
            run_lab(lab=1, task=2)

        def task3():
            run_lab(lab1=1, task=3)


    class Lab2:
        def task1():
            print("GOOD KERNEL")
            run_lab(lab=2, task=1, flags=['-DKERNEL=good'])
            print("===========================================================")
            print("BAD KERNEL")
            run_lab(lab=2, task=1, flags=['-DKERNEL=bad'])

        def task2():
            run_lab(lab=2, task=2, flags=['-DSTREAMS=1'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=2'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=4'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=8'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=16'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=32'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=64'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=128'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=256'])
            run_lab(lab=2, task=2, flags=['-DSTREAMS=512'])

            streams = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
            time = [600.038, 590.494, 582.736, 573.755, 572.416, 567.252, 569.085, 564.892, 562.614, 569.617]

            plt.figure(figsize=(12, 10))
            plt.grid()
            plt.xlabel('streams')
            plt.ylabel('time [ms]')

            plt.plot(streams, time)
            plt.show()

        def task3():
            run_lab(lab=2, task=3, flags=["-D", "OP='c'", "-D", "VERIFY=1", "-D", "SIZE=1024"])
            run_lab(lab=2, task=3, flags=["-D", "OP='g'", "-D", "VERIFY=1", "-D", "SIZE=1024"])
            run_lab(lab=2, task=3, flags=["-D", "OP='s'", "-D", "VERIFY=1", "-D", "SIZE=1024"])

            sizes = 2 ** np.array([4, 6, 8, 10, 12, 14])

            compile(f'lab{2}/task{3}', ["-D", "OP='c'"])
            for s in sizes:
                run(f'lab{2}/task{3}.elf', flags=[str(s)])

            sizes = 2 ** np.array([4, 6, 8, 10, 12, 14])

            compile(f'lab{2}/task{3}', ["-D", "OP='g'"])
            for s in sizes:
                run(f'lab{2}/task{3}.elf', flags=[str(s)])

            sizes = 2 ** np.array([4, 6, 8, 10, 12, 14])

            compile(f'lab{2}/task{3}', ["-D", "OP='s'"])
            for s in sizes:
                run(f'lab{2}/task{3}.elf', flags=[str(s)])

            sizes = 2** np.array([4, 6, 8, 10, 12, 14])

            cpu = [0.003, 0.017, 0.267, 5.328, 324.764, 7213.33]
            glob = [4.2158, 3.4795, 3.2041, 3.5973, 3.2425, 8.55744]
            shar = [4.16973, 3.54509, 3.55034, 3.1552, 3.53005, 7.50848]

            plt.figure(figsize=(8, 6))
            plt.grid()
            plt.xlabel('size [2^x]')
            plt.ylabel('time [ms]')

            #plt.plot(sizes, cpu, label='cpu')
            plt.plot(sizes, glob, label='global')
            plt.plot(sizes, shar, label='shared')

            plt.legend()
            plt.show()


    class Lab3:
        def task1():
            run_lab(lab=3, task=1, flags=["-D", "OP='r'", "-DVERIFY=1"])
            print()
            run_lab(lab=3, task=1, flags=["-D", "OP='l'", "-DVERIFY=1"])

        def task2():
            run_lab(lab=3, task=2)


    class Lab4:
        def task1():
            run_lab(lab=4, task=1)
            convert('./lab4/lena.pgm', './lab4/lena.png')
            convert('./lab4/gpu_blur.pgm', './lab4/gpu_blur.png')
            convert('./lab4/gpu_gauss.pgm', './lab4/gpu_gauss.png')

        def task2():
            run_lab(lab=4, task=2)
            convert('./lab4/lena_scaled.pgm', './lab4/lena_scaled.png')

        def task3():
            run_lab(lab=4, task=3)


def main():
    Lab.Lab1.task1()
    Lab.Lab1.task2()
    Lab.Lab1.task3()

    Lab.lab2.task1()
    Lab.lab2.task2()
    Lab.lab2.task3()

    Lab.lab3.task1()
    Lab.lab3.task2()

    Lab.lab4.task1()
    Lab.lab4.task2()
    Lab.lab4.task3()


if __name__ == '__main__':
    main()

