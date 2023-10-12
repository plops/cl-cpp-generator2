#ifndef GLPROTO_H
#define GLPROTO_H

#include <grpc++/grpc++.h>
#include <proto/gl.grpc.pb.h>
#include <queue>
#include <mutex>
#include <condition_variable> 
class GLProto : public glproto::View::Service {
        public:
        explicit  GLProto ()       ;   
        grpc::Status ClearColor (grpc::ServerContext* context, const glproto::ClearColorRequest* request, glproto::ClearColorReply* reply)       ;   
        private:
        std::mutex queue_mutex_;
        std::condition_variable cv_;
};

#endif /* !GLPROTO_H */