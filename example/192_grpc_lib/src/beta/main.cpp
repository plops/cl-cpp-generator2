#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include <grpcpp/grpcpp.h>
#include "processor.grpc.pb.h"
#include "SensorFusion.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;
using processor::RemoteProcessing;
using processor::ClientMeasurement;
using processor::ServerMessage;
using processor::LogEntry;
using processor::Empty;

class RemoteProcessingImpl final : public RemoteProcessing::Service {
    Status StreamSession(ServerContext* context, 
                         ServerReaderWriter<ServerMessage, ClientMeasurement>* stream) override {
        
        std::cout << "Beta Server: Client connected. Starting session." << std::endl;
        std::vector<Result> results;
        const int num_iterations = 10;

        for (int i = 0; i < num_iterations; ++i) {
            // 1. Send Timestamp
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            auto seconds = std::chrono::duration_cast<std::chrono::seconds>(duration).count();
            auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count() % 1000000000;

            ServerMessage ts_msg;
            auto ts = ts_msg.mutable_timestamp();
            ts->mutable_time()->set_seconds(seconds);
            ts->mutable_time()->set_nanos(nanos);
            ts->set_id(i);

            Timestamp current_ts{static_cast<uint64_t>(seconds), static_cast<uint32_t>(nanos)};

            if (!stream->Write(ts_msg)) {
                return Status::CANCELLED;
            }

            // 2. Read Measurement
            ClientMeasurement measurement;
            if (!stream->Read(&measurement)) {
                break;
            }

            // 3. Compute and Send Result
            Measurement m{measurement.value(), measurement.timestamp_id()};
            Result r = SensorFusion::fuse(m, current_ts);
            results.push_back(r);

            ServerMessage res_msg;
            auto res = res_msg.mutable_result();
            res->set_fused_value(r.fused_value);
            res->set_timestamp_id(r.timestamp_id);
            res->set_session_ended(false);

            if (!stream->Write(res_msg)) {
                return Status::CANCELLED;
            }
        }

        // Final summary
        ServerMessage final_msg;
        auto final_res = final_msg.mutable_result();
        final_res->set_session_ended(true);
        final_res->set_summary(SensorFusion::generate_summary(results));
        stream->Write(final_msg);

        std::cout << "Beta Server: Session ended." << std::endl;
        return Status::OK;
    }

    Status SubmitLogs(ServerContext* context, grpc::ServerReader<LogEntry>* reader, Empty* response) override {
        LogEntry entry;
        while (reader->Read(&entry)) {
            std::cout << "[CLIENT LOG] [" << entry.level() << "] " << entry.message() << std::endl;
        }
        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    RemoteProcessingImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Beta Server listening on " << server_address << std::endl;
    server->Wait();
}

int main(int argc, char** argv) {
    RunServer();
    return 0;
}
