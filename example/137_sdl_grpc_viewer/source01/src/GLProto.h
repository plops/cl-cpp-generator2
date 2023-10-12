#ifndef GLPROTO_H
#define GLPROTO_H

#include <grpc++/grpc++.h>
#include <proto/gl.grpc.pb.h>
#include <queue>
#include <mutex>
#include <condition_variable> 
class GLProto : public glproto::Service {
        public:
        explicit  GLProto ()       ;   
        defmethod;
        ClearColor
        context(request, reply);
        declare(type(grpc::ServerContext*, context), type(ClearColorRequest*, request), type(ClearColorReply*, reply), values(grpc::Status));
        return grpc::Status::OK;
        private:
        std::queue<Command> command_queue_;
        std::mutex queue_mutex_;
        std::condition_variable cv_;
};

#endif /* !GLPROTO_H */