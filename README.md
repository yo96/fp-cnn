# fp-cnn
End-to-end frame work to map a CNN onto FPGA.

## How to use the makefile
- Software emulation   
~~~
make -f sdaccel.mk run_cpu_em
~~~

- Hardware emulation   
~~~
make -f sdaccel.mk run_hw_emu
~~~

- Run on board   
~~~
make -f sdaccel.mk run_hw
~~~

- Clean   
~~~
make -f sdaccel.mk cleanall
~~~
