// no preamble

#include "CallData.h"
#include <iostream>
#include <stdexcept>
CallData::CallData(glproto::View::AsyncService *service,
                   grpc::ServerCompletionQueue *cq,
                   const grpc::ServerContext &ctx)
    : service_(service), cq_(cq), ctx_(ctx), responder_(ctx), status_(CREATE) {
  Proceed();
}
void CallData::Proceed() {
  if (CREATE == status_) {
    status_ = PROCESS;
    service_->RequestClearColor(&ctx_, &responder_, cq_, cq_, this);
  } else if (PROCESS == status_) {
    new CallData(service_, cq_);
    reply_.set_success(true);
    status_ = FINISH;
    responder_.Finish(reply_, grpc::Status::OK, this);
  } else {
    GPR_ASSERT(status_ == FINISH);
    delete (this);
  }
}