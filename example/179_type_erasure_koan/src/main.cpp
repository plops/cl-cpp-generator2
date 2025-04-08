#include <iostream>

using namespace std;

// type erasure using inheritance
template <typename T>
class smartptr {
private:
    class destroy_base {
    public:
        virtual void operator()(void*) = 0;
        virtual      ~destroy_base() = default;
    };

    template <typename Deleter>
    class destroy : public destroy_base {
    public:
        destroy(Deleter d) :
            d_{d} {}

        void operator()(void* p) override { d_(static_cast<T*>(p)); }

    private:
        Deleter d_;
    };

public:
    template <typename Deleter>
    smartptr(T* p, Deleter d) :
        p_{p}, d_{new destroy<Deleter>(d)} {
        // cout << "smartptr" << endl;
    }

    ~smartptr() {
        // cout << "~smartptr" << endl;
        (*d_)(p_);
        delete d_;
    }

    T*       operator->() { return p_; }
    const T* operator->() const { return p_; }

private:
    T*            p_;
    destroy_base* d_;
};

// local buffer optimization
template <typename T>
class smartptr_lbo {
private:
    class destroy_base {
    public:
        virtual void operator()(void*) = 0;
        virtual      ~destroy_base() = default;
    };

    template <typename Deleter>
    class destroy : public destroy_base {
    public:
        destroy(Deleter d) :
            d_{d} {}

        void operator()(void* p) override { d_(static_cast<T*>(p)); }

    private:
        Deleter d_;
    };

public:
    template <typename Deleter>
    smartptr_lbo(T* p, Deleter d) :
        p_{p} {
        static_assert(sizeof(Deleter) <= sizeof(buf_));
        ::new(static_cast<void*>(buf_)) destroy<Deleter>(d);
        cout << "smartptr_lbo" << endl;
    }

    ~smartptr_lbo() {
        cout << "~smartptr_lbo" << endl;
        destroy_base* d = reinterpret_cast<destroy_base*>(buf_);
        (*d)(p_);
        d->~destroy_base();
    }

    T*       operator->() { return p_; }
    const T* operator->() const { return p_; }

private:
    T*              p_;
    alignas(8) char buf_[16];
};


int main(int argc, char* argv[]) {
    struct Point {
        int   x;
        float y;
    };

    {
        smartptr_lbo<Point> p(new Point({1, 2.3}), [](void* p) {
            cout << "smartptr_lbo deleter lambda" << endl;
            // delete p;
        });
        cout << p->x << " " << p->y << endl;
        auto q = p;
        q->x   = 3;
        cout << q->x << " " << q->y << endl;
        cout << p->x << " " << p->y << endl;
    }

    {
        smartptr<Point> p(new Point({1, 2.3}), [](void* p) {
            cout << "smartptr deleter lambda" << endl;
            // delete p;
        });
        cout << p->x << " " << p->y << endl;
        auto q = p;
        q->x   = 3;
        cout << q->x << " " << q->y << endl;
        cout << p->x << " " << p->y << endl;
    }



    return 0;
}
