// no preamble

#include "Physics.h"
#include <box2d/box2d.h>
#include <iostream>
#include <stdexcept>
Physics::Physics()
    : time_step_(1.6666667e-2F), velocity_iterations_(6),
      position_iterations_(2), gravity_(b2Vec2(0.F, -10.F)),
      world_(b2World(gravity_)), body_(nullptr) {
  // https://github.com/erincatto/box2d/blob/main/unit-test/hello_world.cpp

  auto groundBodyDef = b2BodyDef();
  groundBodyDef.position.Set(0.F, -10.F);
  auto *groundBody = world_.CreateBody(&groundBodyDef);
  auto groundBox = b2PolygonShape();
  groundBox.SetAsBox(50.F, 10.F);
  groundBody->CreateFixture(&groundBox, 0.F);
  auto bodyDef = b2BodyDef();
  bodyDef.type = b2_dynamicBody;
  bodyDef.position.Set(0.F, 4.0F);
  body_ = world_.CreateBody(&bodyDef);
  auto dynamicBox = b2PolygonShape();
  dynamicBox.SetAsBox(1.0F, 1.0F);
  auto fixtureDef = b2FixtureDef();
  fixtureDef.shape = &dynamicBox;
  fixtureDef.density = 1.0F;
  fixtureDef.friction = 0.30F;
  body_->CreateFixture(&fixtureDef);
}
std::tuple<float, float, float> Physics::Step() {
  world_.Step(time_step_, velocity_iterations_, position_iterations_);
  auto position = body_->GetPosition();
  auto angle = body_->GetAngle();
  std::cout << ""
            << " position.x='" << position.x << "' "
            << " position.y='" << position.y << "' "
            << " angle='" << angle << "' " << std::endl;
  return std::make_tuple(position.x, position.y, angle);
}