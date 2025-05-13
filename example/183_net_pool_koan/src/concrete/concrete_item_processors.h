//
// Created by martin on 5/13/25.
//

#ifndef CONCRETE_ITEM_PROCESSORS_H
#define CONCRETE_ITEM_PROCESSORS_H

#include "src/interfaces/iitem_processor.h"
#include "src/common/common.h"
#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <random> // For rand() in sleep

class LoggingImageProcessor : public IItemProcessor<Image> {
public:
    void process(const Image& image, std::size_t item_index) override {
        std::byte first_byte = image.empty() ? std::byte{0} : image[0];
        std::cout << "Proc Img " << item_index << ": Size=" << image.size()
                  << ", FirstByte=" << std::to_integer<int>(first_byte) << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(10 + (rand() % 20)));
    }
};

class LoggingMetadataProcessor : public IItemProcessor<Metadata> {
public:
    void process(const Metadata& metadata, std::size_t item_index) override {
        std::cout << "Proc Meta " << item_index << ": i=" << metadata.i << ", f=" << metadata.f << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(5 + (rand() % 10)));
    }
};

class LoggingMeasurementProcessor : public IItemProcessor<Measurement> {
public:
    void process(const Measurement& measurement, std::size_t item_index) override {
        std::cout << "Proc Meas " << item_index << ": q=" << measurement.q << ", p=" << measurement.p << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(2 + (rand() % 5)));
    }
};
#endif //CONCRETE_ITEM_PROCESSORS_H
