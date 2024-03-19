run: main.cc libs/PlanBaseVisitor.cpp libs/PlanLexer.cpp libs/PlanParser.cpp libs/PlanVisitor.cpp proto/common.pb.cc proto/Plan.pb.cc proto/schema.pb.cc 
	g++ -o $@ -Wl,--copy-dt-needed-entries -std=gnu++17 -I/usr/include/antlr4-runtime -I./libs -I./proto $^ -lantlr4-runtime -lprotobuf -g

all: run
	./$< "x in [1,2,3]"

clean: 
	rm run
