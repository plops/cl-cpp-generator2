#pragma once
template <class Lambda, class... Args> class QAppLambda : public AnyQAppLambda {
public:
  Lambda lambda;
  std::tuple<Args...> args;
  QAppLambda(Lambda lambda, Args... args)
      : AnyQAppLambda{}, lambda{lambda}, args{std::make_tuple(args...)} {}
  void run() override { run_impl(std::make_index_sequence<sizeof...(Args)>()); }
  template <std::size_t... I> void run_impl(std::index_sequence<I...>) {
    lambda(std::get<I>(args)...);
  }
};