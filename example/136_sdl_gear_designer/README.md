# Overview Box2D library

  **Box2D 2.4.1**: A 2D rigid body simulation library written in portable C++ for game development.
  
  **Prerequisites**:
  - Familiarity with basic physics concepts (mass, force, torque, impulses).
  - Prior experience in C++ programming, compiling, and debugging.

  **Core Concepts**:
  - Shape: 2D geometrical objects like circles or polygons.
  - Rigid Body: Inflexible matter with constant inter-particle distances.
  - Fixture: Binds shapes to rigid bodies and handles collision properties.
  - Constraint: Limits degrees of freedom in rigid bodies.
  - Joint: Holds two or more bodies together.
  - World: Collection of bodies, fixtures, and constraints.
  - Solver: Resolves time advancement, contact, and joint constraints.

  **Modules**:
  - Common: Code for allocation, math, and settings.
  - Collision: Defines shapes and handles collision queries.
  - Dynamics: Manages the simulation world, bodies, and joints.

  **Units**:
  - Uses MKS (meters, kilograms, seconds) units.
  - Recommended object sizes between 0.1 and 10 meters.
  - World size limited to less than 2 kilometers for best results.

  **Customization and Factories**:
  - Allows unit changes via `b2_lengthUnitsPerMeter`.
  - Memory management via factory methods (`CreateBody`, `DestroyBody`, `CreateJoint`, `DestroyJoint`).

  **Additional Info**:
  - Updates only with new releases.
  - File bugs and feedback on Box2D Issues.
  - Supports Discord and subreddit communities.
  
  **Caution Points**:
  - Not suitable for C++ beginners.
  - Don't use pixels for units.
  - Limit world sizes for better performance.
