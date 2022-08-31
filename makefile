#!/bin/bash
EXE	= mbo
CC	= g++
SRC	= ./src
BIN	= ./bin

all: $(BIN)/$(EXE)
 
$(BIN)/$(EXE): 
	$(CC) -O3 -o $(BIN)/$(EXE) $(SRC)/mbo.cpp -pthread

