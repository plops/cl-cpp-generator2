# Introduction

Here I'm learning about type erasure. This is a programming pattern
that helps with decoupling dependencies.

A diagram that shows what kind of issues this solves is shown in this
short talk:
'MUC++_Klaus_Iglberger_-_A_Brief_Introduction_to_Type_Erasure-[SPelQUPcHQQ]'
at minute 8:00.




# References

## Youtube video 1 
- There_is_no_silver_bullet_-_Klaus_Iglberger_-_Meeting_C++_2024
- 5F3d3LocM3k

keywords: value-based object oriented programming, Progressive C++,
type erasure

**No Silver Bullet: Exploring Design Choices in Modern C++**

*   **0:00:08 Introduction:** Claus introduces the topic of design
    choices in modern C++, focusing on the shift away from traditional
    object-oriented patterns.
*   **0:00:33 The Problem with Inheritance:** The traditional approach
    using inheritance and virtual functions leads to tight coupling
    and maintenance issues, especially in large codebases.
*   **0:09:09 Modern C++ Isn't Always the Answer:** Modern C++
    features like unique pointers and move semantics don't
    automatically solve design problems associated with inheritance.
*   **0:13:19 Variant-Based Solution:** A variant-based approach
    offers a more functional style, eliminating inheritance and
    pointers for improved simplicity and potential performance gains.
*   **0:22:29 Architectural Issues with Variants:** Using variants
    directly can lead to tight coupling and hinder code
    maintainability, as it exposes implementation details and
    complicates extensibility.
*   **0:25:30 Templates as a Solution:** Function templates can help
    invert dependencies and improve code modularity when working with
    variants.
*   **0:27:36 The Template Trap:** Overusing templates can result in
    excessive code bloat and increased compile times, especially in
    large projects.
*   **0:28:32 The Spectrum of Design Choices:** Variants and virtual
    functions represent opposite ends of a design spectrum, each with
    its own strengths and weaknesses.
*   **0:30:17 The Importance of Design:** Good software design, rather
    than specific language features, is crucial for project
    success. The talk emphasizes the need for a stronger focus on
    design principles in the C++ community.
*   **0:36:26 Progressive C++:** The concept of "Progressive C++" is
    introduced as a new term to describe modern C++ development that
    emphasizes design philosophy over specific language features.
*   **0:36:38 Type Erasure as a Solution:** Type erasure, using a
    combination of inheritance and templates hidden behind a
    value-like interface, offers a balance between flexibility and
    maintainability.
*   **0:47:25 Performance Considerations:** The type erasure approach
    can offer performance benefits, especially when combined with
    small buffer optimizations.
*   **0:49:16 Choosing the Right Tool:** The choice between a
    variant-based (functional) and a type erasure-based
    (object-oriented) approach depends on the specific architectural
    needs of the project.
*   **0:51:51 Reflection and the Future:** C++ reflection, expected in
    future standards, may offer new possibilities for generating type
    erasure wrappers and potentially simplifying or replacing
    variant-based solutions.
*   **0:53:24 Conclusion:** There is no one-size-fits-all solution in
    software design. Understanding the trade-offs between different
    approaches and prioritizing architectural considerations are
    essential for building maintainable and efficient software.
*   **0:54:16 Q&A:** Using libraries to reduce boilerplate from type
    erasure and using C++ concepts with a functional approach are
    discussed.
*   **0:56:43 CRTP is not a replacement:** CRTP is not a general
    solution for replacing inheritance hierarchies due to its
    limitations in terms of type relationships and its tendency to
    push designs towards a template-heavy approach.

## Youtube Video 2

MUC++_Nicolas_Chausse_-_Two_Kinds_of_Type_Erasure-[QgMQPqVc6JE]
