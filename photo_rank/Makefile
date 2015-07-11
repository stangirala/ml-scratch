app_name = photo_rank

main_file = photo_rank.cpp

clean:
	/bin/rm target/$(app_name)

all:
	g++ -std=c++11 $(main_file) -lopencv_highgui -lopencv_core -lopencv_imgproc -o target/$(app_name)
