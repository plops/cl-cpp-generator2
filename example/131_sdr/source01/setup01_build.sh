# https://news.ycombinator.com/item?id=35758898
# https://nullprogram.com/blog/2023/04/29/

g++ \
    main.cpp \
    -lSoapySDR \
    -g -g3 -ggdb -gdwarf-4 \
    -Wall -Wextra -Wconversion -Wdouble-promotion \
    -Wno-unused-parameter -Wno-unused-function -Wno-sign-conversion \
    -fsanitize=undefined  -fanalyzer \
    -Wvla -Wframe-larger-than=5000 -Wstack-usage=10000 \
    -Wshadow -Werror \
    -fvisibility=hidden \
    -fno-strict-overflow -Wno-strict-overflow
