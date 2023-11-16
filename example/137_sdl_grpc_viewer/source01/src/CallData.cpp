// no preamble

#include "CallData.h"
#include <iostream>
#include <stdexcept>
CallData::CallData(glproto::View::AsyncService *service,
                   grpc::ServerCompletionQueue *cq,
                   const grpc::ServerContext &ctx)
    : service_(service), cq_(cq), ctx_(ctx), status_(CREATE) {
  Proceed();
}
void CallData::Proceed() {}