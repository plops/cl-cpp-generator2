//
// Created by martin on 5/13/25.
//

#ifndef APPLICATION_H
#define APPLICATION_H

#include "src/common/common.h"
#include "src/core/data_pool.h"
#include "src/concrete/simulated_network_receiver.h"
#include "src/concrete/concrete_item_processors.h"
#include "src/core/producer.h"
#include "src/core/consumer.h"

#include <vector>
#include <thread> // jthread
#include <memory> // unique_ptr
#include <atomic>
#include <csignal>
#include <iostream>

class Application {
public:
    Application() : shutdown_requested_(false) {
        image_pool_storage_ = std::make_unique<DataPool<Image>>(IMAGE_POOL_SIZE);
        metadata_pool_storage_ = std::make_unique<DataPool<Metadata>>(METADATA_POOL_SIZE);
        measurement_pool_storage_ = std::make_unique<DataPool<Measurement>>(MEASUREMENT_POOL_SIZE);

        network_receiver_ = std::make_unique<SimulatedNetworkReceiver>();
        image_processor_ = std::make_unique<LoggingImageProcessor>();
        metadata_processor_ = std::make_unique<LoggingMetadataProcessor>();
        measurement_processor_ = std::make_unique<LoggingMeasurementProcessor>();
        std::cout << "Application components initialized." << std::endl;
    }

    void run() {
        std::cout << "Application starting run..." << std::endl;
        signal(SIGINT, Application::signal_handler_static);
        signal(SIGTERM, Application::signal_handler_static);
        s_application_instance = this;

        threads_.emplace_back(producer_task, std::ref(*network_receiver_),
                              std::ref(*image_pool_storage_), std::ref(*metadata_pool_storage_),
                              std::ref(*measurement_pool_storage_));
        threads_.emplace_back(consumer_task<Image>, "Consumer Image",
                              std::ref(*image_pool_storage_), std::ref(*image_processor_));
        threads_.emplace_back(consumer_task<Metadata>, "Consumer Metadata",
                              std::ref(*metadata_pool_storage_), std::ref(*metadata_processor_));
        threads_.emplace_back(consumer_task<Measurement>, "Consumer Measurement",
                              std::ref(*measurement_pool_storage_), std::ref(*measurement_processor_));
        std::cout << "Threads started." << std::endl;

        while (!shutdown_requested_.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        shutdown();
        std::cout << "Application run loop finished." << std::endl;
    }

private:
    void shutdown() {
        std::cout << "Initiating shutdown sequence..." << std::endl;
        for (auto& t : threads_) {
            if (t.joinable()) t.request_stop();
        }
        std::cout << "Stop requested for all threads." << std::endl;
        if (network_receiver_) network_receiver_->stop();
        std::cout << "Network receiver stopped." << std::endl;
        if (image_pool_storage_) image_pool_storage_->stop_all();
        if (metadata_pool_storage_) metadata_pool_storage_->stop_all();
        if (measurement_pool_storage_) measurement_pool_storage_->stop_all();
        std::cout << "Pools stopped." << std::endl;
        std::cout << "Waiting for threads to join..." << std::endl;
        threads_.clear();
        std::cout << "All threads joined." << std::endl;
    }

    static Application* s_application_instance;
    static void signal_handler_static(int signum) {
        if (s_application_instance) {
            std::cout << "\nSignal " << signum << " received." << std::endl;
            s_application_instance->shutdown_requested_.store(true, std::memory_order_relaxed);
        }
    }

    std::atomic<bool> shutdown_requested_;
    std::vector<std::jthread> threads_;
    std::unique_ptr<INetworkReceiver> network_receiver_;
    std::unique_ptr<IItemProcessor<Image>> image_processor_;
    std::unique_ptr<IItemProcessor<Metadata>> metadata_processor_;
    std::unique_ptr<IItemProcessor<Measurement>> measurement_processor_;
    std::unique_ptr<DataPool<Image>> image_pool_storage_;
    std::unique_ptr<DataPool<Metadata>> metadata_pool_storage_;
    std::unique_ptr<DataPool<Measurement>> measurement_pool_storage_;
};
Application* Application::s_application_instance = nullptr;

#endif //APPLICATION_H
