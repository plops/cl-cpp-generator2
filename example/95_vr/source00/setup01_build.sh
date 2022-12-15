#!/bin/bash
export PATH=$PATH:/home/martin/quest2/ndk/android-ndk-r25b/toolchains/llvm/prebuilt/linux-x86_64/bin
export PATH=$PATH:/usr/lib/jvm/java-1.8.0-openjdk-1.8.0.352.b08-2.fc37.x86_64/bin/
export PATH=$PATH:


javac\
	-classpath $ANDROID_HOME/platforms/android-26/android.jar\
	-d .\
	../src/main/java/com/makepad/hello_quest/*.java
