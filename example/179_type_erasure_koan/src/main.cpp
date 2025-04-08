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
        cout << "smartptr" << endl;
    }

    ~smartptr() {
        cout << "~smartptr" << endl;
        (*d_)(p_);
        delete d_;
    }

    T*       operator->() { return p_; }
    const T* operator->() const { return p_; }

private:
    T*            p_;
    destroy_base* d_;
};


int main(int argc, char* argv[]) {
    struct Point {int x; float y;};
    smartptr<Point> p(new Point({1,2.3}), [](void*p){delete p;});
    cout << p->x << " " << p->y << endl;
    return 0;
}
