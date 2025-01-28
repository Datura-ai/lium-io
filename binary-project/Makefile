all: main

main: main.cpp
	g++ -o system_monitor main.cpp -I/usr/include/botan-2 /usr/local/lib/libbotan-2.a -ldl -lpthread -lssl -lcrypto -lbotan-2 -std=c++17

clean:
	rm -f main