CC=gcc
CFLAGS=-ansi -g
DBG = clang

cs240StreamingService: main.c streaming_service.h
	$(CC) $(CFLAGS) $< -o $@ streaming_service.c

.PHONY: clean

clean:
	rm -f cs240StreamingService