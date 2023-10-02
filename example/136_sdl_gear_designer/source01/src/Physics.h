#ifndef PHYSICS_H
#define PHYSICS_H

#include <box2d/box2d.h> 
class Physics  {
        public:
        explicit  Physics ()       ;   
        std::tuple<float,float,float> Step ()       ;   
        private:
        float time_step_;
        int velocity_iterations_;
        int position_iterations_;
        b2Vec2 gravity_;
        b2World world_;
        b2Body* body_;
};

#endif /* !PHYSICS_H */