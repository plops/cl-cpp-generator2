#pragma once
#include <boost/container/allocator_traits.hpp>
/*
  The MIT License (MIT)

  Copyright (c) 2016-2017 David Gross
*/

extern "C" {
//#define PAPI_VERSION_NUMBER(maj,min,rev) (((maj)<<16) | ((min)<<8) | (rev))
/* This is the PAPI version on which we are running */
//#define PAPI_VERSION  			PAPI_VERSION_NUMBER(3,0,6)
#include "/home/martin/vulkan/include/papi.h"
//#include "/home/martin/src/papi-7.2.0b1/src/papivi.h"
}

#include <cstddef>
#include <array>
#include <string>
#include <stdexcept>

//#define likely_true(x)   __builtin_expect(!!(x), 1)
#define likely_false(x)  __builtin_expect(!!(x), 0)

namespace papi
{
    using event_code = int;
    using papi_counter = long long;

    inline std::string get_event_code_name(event_code code)
    {
        std::array<char, PAPI_MAX_STR_LEN> event_name;
        ::PAPI_event_code_to_name(code, event_name.data());

        return event_name.data();
    }

    template <event_code _Event>
    struct event
    {
        explicit event(papi_counter counter = {})
            : _counter(counter)
        {
        }

        papi_counter counter() const { return _counter; }

        static constexpr event_code code() { return _Event; }
        static const std::string& name() { return s_name; }

    private:
        static const std::string s_name;

        papi_counter _counter;
    };

    template <event_code _Event>
    const std::string event<_Event>::s_name = get_event_code_name(_Event);

    template <typename _Stream, event_code _Event>
    inline _Stream& operator<<(_Stream& strm, const event<_Event>& evt)
    {
        strm << evt.name() << "=" << evt.counter();
        return strm;
    }

    template <event_code... _Events>
    struct event_set
    {
        explicit event_set()
        {
            if (!_init)
            {
                if (auto retval = PAPI_library_init(PAPI_VER_CURRENT); retval != PAPI_VER_CURRENT)
                    throw std::runtime_error("PAPI initialization failed");
                /* Initialize the EventSet */
                if (auto retval = PAPI_create_eventset(&_event_set); retval != PAPI_OK)
                    throw std::runtime_error("PAPI create eventset failed");
                _init = true;
            }
        }

        void start_counters()
        {
            int ret;

            if (likely_false((ret = ::PAPI_start(_event_set)) != PAPI_OK))
                throw std::runtime_error(std::string("PAPI_start_counters failed with error: ") + PAPI_strerror(ret));
        }

        void reset_counters()
        {
            int ret;

             if (likely_false((ret = ::PAPI_reset(_event_set)) != PAPI_OK))
                 throw std::runtime_error(std::string("PAPI_reset failed with error: ") + PAPI_strerror(ret));
        }

        void stop_counters()
        {
            int ret;
            if (likely_false((ret = ::PAPI_stop(_event_set, _counters.data())) != PAPI_OK))
                throw std::runtime_error(std::string("PAPI_stop_counters failed with error: ") + PAPI_strerror(ret));
        }

        static constexpr std::size_t size() { return sizeof...(_Events); }
        //static_assert(size() > 0, "at least one hardware event has to be in the set");

        template <std::size_t _EventIndex>
        auto at() const
        {
            static constexpr const std::array<event_code, sizeof...(_Events)> events = {{_Events...}};
            constexpr event_code code = events[_EventIndex];
            return event<code>(_counters[_EventIndex]);
        }

        template <event_code _EventCode>
        auto get() const
        {
            static constexpr const std::array<event_code, sizeof...(_Events)> events = {{_Events...}};

            constexpr int eventIndex = find(_EventCode, events, sizeof...(_Events), 0);
            static_assert(eventIndex != -1, "EventCode not present in this event_set");
            return at<eventIndex>();
        }

    private:
        bool _init{false};
        int _event_set{PAPI_NULL};

        template <typename ArrayT>
        static constexpr int find(event_code x, ArrayT& ar, std::size_t size, std::size_t i)
        {
            return size == i ? -1 : (ar[i] == x ? i : find(x, ar, size, i + 1));
        }

        static std::array<event_code, sizeof...(_Events)> s_events;

        std::array<papi_counter, sizeof...(_Events)> _counters;
    };

    template <event_code... _Events>
    std::array<event_code, sizeof...(_Events)> event_set<_Events...>::s_events = {{_Events...}};

    namespace detail
    {
        template <std::size_t N, typename _Stream, event_code... _Events>
        inline std::enable_if_t<N == event_set<_Events...>::size()>
        to_stream(_Stream&, const event_set<_Events...>&)
        {
        }

        template <std::size_t N, typename _Stream, event_code... _Events>
        inline std::enable_if_t<N < event_set<_Events...>::size()>
        to_stream(_Stream& strm, const event_set<_Events...>& set)
        {
            strm << set.template at<N>() << " ";
            detail::to_stream<N + 1>(strm, set);
        }
    }

    template <typename _Stream, event_code... _Events>
    inline _Stream& operator<<(_Stream& strm, const event_set<_Events...>& set)
    {
        detail::to_stream<0>(strm, set);
        return strm;
    }
}
