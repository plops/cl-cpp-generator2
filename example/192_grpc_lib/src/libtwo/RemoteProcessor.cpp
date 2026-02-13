#include "RemoteProcessor.hpp"
#include <iostream>

RemoteProcessor::RemoteProcessor() {
    channel_ = grpc::CreateChannel("localhost:50051", grpc::InsecureChannelCredentials());
    stub_ = processor::RemoteProcessing::NewStub(channel_);
}

void RemoteProcessor::start_session(
    MeasurementProvider measure_prov,
    ResultReceiver result_recv,
    SummaryReceiver summary_recv
) {
    grpc::ClientContext context;
    auto stream = stub_->StreamSession(&context);

    processor::ServerMessage server_msg;
    while (stream->Read(&server_msg)) {
        if (server_msg.has_timestamp()) {
            const auto& p_ts = server_msg.timestamp();
            Timestamp ts{p_ts.time().seconds(), p_ts.time().nanos()};
            
            // Get measurement from Alpha
            Measurement m = measure_prov(ts);
            
            processor::ClientMeasurement cm;
            cm.set_value(m.value);
            cm.set_timestamp_id(m.timestamp_id);
            
            if (!stream->Write(cm)) {
                break;
            }
        } else if (server_msg.has_result()) {
            const auto& p_res = server_msg.result();
            
            if (p_res.session_ended()) {
                summary_recv(p_res.summary());
                break;
            } else {
                Result r{p_res.fused_value(), p_res.timestamp_id()};
                result_recv(r);
            }
        }
    }
    stream->WritesDone();
    grpc::Status status = stream->Finish();
    if (!status.ok()) {
        std::cerr << "RemoteProcessor: Session stream failed: " << status.error_message() << std::endl;
    }
}

void RemoteProcessor::ensure_log_stream() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (!log_enabled_) {
        log_stream_ = stub_->SubmitLogs(&log_context_, &log_response_);
        log_enabled_ = true;
    }
}

void RemoteProcessor::log(const std::string& message) {
    ensure_log_stream();
    processor::LogEntry entry;
    entry.set_message(message);
    entry.set_level("INFO");
    
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_enabled_) {
        if (!log_stream_->Write(entry)) {
            log_enabled_ = false;
        }
    }
}

extern "C" IProcessor* create_processor() {
    return new RemoteProcessor();
}
