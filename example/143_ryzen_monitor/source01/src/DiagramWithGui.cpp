// no preamble

#include "DiagramWithGui.h"
#include "implot.h"
#include <format>
void DiagramWithGui::RenderGui(bool xticks) {
  struct PlotData {
    const std::deque<float> &time_points_;
    const std::vector<DiagramData> &diagrams_;
    int i;
    PlotData(const std::deque<float> &time_points,
             const std::vector<DiagramData> &diagrams, int index)
        : time_points_(time_points), diagrams_(diagrams), i(index) {}
  };
  if (ImPlot::BeginPlot(name_y_.c_str(), ImVec2(-1, 130),
                        ImPlotFlags_NoFrame | ImPlotFlags_NoTitle)) {
    for (auto i = 0; i < max_cores_; i += 1) {
      auto data{PlotData(time_points_, diagrams_, i)};
      auto getter{[](int idx, void *data_) -> ImPlotPoint {
        const auto d{static_cast<PlotData *>(data_)};
        auto x{d->time_points_.at(idx)};
        auto y{d->diagrams_.at(d->i).values.at(idx)};
        return ImPlotPoint(x, y);
      }};
      ImPlot::SetupAxes("X", name_y_.c_str(),
                        ImPlotAxisFlags_AutoFit | ImPlotAxisFlags_NoLabel |
                            (xticks ? 0 : ImPlotAxisFlags_NoTickLabels),
                        ImPlotAxisFlags_AutoFit);
      ImPlot::PlotLineG(std::format("Core {:2}", i).c_str(), getter,
                        static_cast<void *>(&data),
                        static_cast<int>(time_points_.size()));
    }
    ImPlot::EndPlot();
  }
}