#ifndef CALLDATA_H
#define CALLDATA_H

#include <grpc++/grpc++.h>
#include <proto/gl.grpc.pb.h> 
class CallData : public glproto::View::Service {
        public:
        enum CallStatus {CREATE, PROCESS, FINISH};
        explicit  CallData (glproto::View::AsyncService* service, grpc::ServerCompletionQueue* cq, const grpc::ServerContext & ctx)       ;   
        void Proceed ()       ;   
        private:
        glproto::View::AsyncService* service_;
        grpc::ServerCompletionQueue* cq_;
        const grpc::ServerContext & ctx_;
        glproto::ClearColorRequest request_;
        glproto::ClearColorReply reply_;
        CallStatus status_;
};

#endif /* !CALLDATA_H */