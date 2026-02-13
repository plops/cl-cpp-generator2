#pragma once
#include "interfaces/IProcessor.hpp"

class LocalProcessor : public IProcessor {
public:
    LocalProcessor();
    virtual ~LocalProcessor() = default;

    void start_session(
        MeasurementProvider measure_prov,
        ResultReceiver result_recv,
        SummaryReceiver summary_recv
    ) override;

    void log(const std::string& message) override;
};
