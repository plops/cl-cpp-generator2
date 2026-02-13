#pragma once
#include "interfaces/IProcessor.hpp"
#include <grpcpp/grpcpp.h>
#include "processor.grpc.pb.h"
#include <thread>
#include <mutex>

class RemoteProcessor : public IProcessor {
public:
    RemoteProcessor();
    virtual ~RemoteProcessor() = default;

    void start_session(
        MeasurementProvider measure_prov,
        ResultReceiver result_recv,
        SummaryReceiver summary_recv
    ) override;

    void log(const std::string& message) override;

private:
    std::unique_ptr<processor::RemoteProcessing::Stub> stub_;
    std::shared_ptr<grpc::Channel> channel_;
    
    // Logging stream
    std::unique_ptr<grpc::ClientWriter<processor::LogEntry>> log_stream_;
    processor::Empty log_response_;
    grpc::ClientContext log_context_;
    std::mutex log_mutex_;
    bool log_enabled_ = false;

    void ensure_log_stream();
};
