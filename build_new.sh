#!/bin/sh

cmake -B build_new -DCSOT_CACHE_SIM_SRC=cache_sim_new.cpp && cmake --build build_new -j
