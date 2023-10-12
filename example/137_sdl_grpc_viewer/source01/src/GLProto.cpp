// no preamble

#include "GLProto.h"
#include <iostream>
#include <stdexcept>
GLProto::GLProto() {}
grpc::Status GLProto::ClearColor(grpc::ServerContext *context,
                                 const glproto::ClearColorRequest *request,
                                 glproto::ClearColorReply *reply) {
  std::cout << std::format(
      "clear-color request->red()='{}' request->green()='{}' "
      "request->blue()='{}' request->alpha()='{}'\n",
      request->red(), request->green(), request->blue(), request->alpha());
  return grpc::Status::OK;
}