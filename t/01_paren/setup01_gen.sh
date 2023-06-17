cd /home/martin/stage/cl-cpp-generator2/t/01_paren/source00
rm *.cpp
cd /home/martin/stage/cl-cpp-generator2/t/01_paren
sbcl --load gen00.lisp --eval '(sb-ext:quit)'
cd /home/martin/stage/cl-cpp-generator2/t/01_paren/source00/b
rm c??_*
./setup01_compile.sh
./setup02_run.sh
cat ../strings.txt
