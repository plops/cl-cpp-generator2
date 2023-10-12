// no preamble

#include "GLProto.h"
#include <iostream>
#include <stdexcept>
GLProto::GLProto() : command_queue_(0), queue_mutex_(0), cv_(0) {}